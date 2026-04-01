"""
评估工具 — A/B 对比 + 统计显著性 p-value + 多指标 IR 评估

用法:
    # 对比两个系统的评估结果
    python -m app.rag.eval compare run_a.json run_b.json --metric ndcg@10

    # 单个系统 IR 指标计算（从 qrels + run 文件）
    python -m app.rag.eval ir-metrics run.json qrels.jsonl

    # 仅做 p-value 检验
    python -m app.rag.eval significance scores_a.json scores_b.json
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Permutation Test / P-Value
# ============================================================

def permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    metric: str = "mean",
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> Dict:
    """双尾置换检验 — 比较 A/B 两个系统的性能差异是否显著

    Args:
        scores_a: 系统 A 的每个查询的指标值
        scores_b: 系统 B 的每个查询的指标值
        metric: 聚合方法 ("mean")
        n_permutations: 置换次数
        seed: 随机种子

    Returns:
        {"A_mean", "B_mean", "Diff", "p_value", "significant", ...}
    """
    rng = random.Random(seed)

    assert len(scores_a) == len(scores_b), (
        f"score lists must be same length: {len(scores_a)} vs {len(scores_b)}"
    )

    n = len(scores_a)
    if n == 0:
        return {
            "n_queries": 0,
            "A_mean": 0.0, "B_mean": 0.0,
            "Diff": 0.0, "p_value": 1.0,
            "significant": False, "alpha": 0.05,
        }

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n
    obs_diff = mean_a - mean_b

    pairs = list(zip(scores_a, scores_b))

    # 置换分布
    extreme_count = 0
    for _ in range(n_permutations):
        # 随机交换每个 pair 的方向
        perm_a, perm_b = [], []
        for sa, sb in pairs:
            if rng.random() < 0.5:
                perm_a.append(sb)
                perm_b.append(sa)
            else:
                perm_a.append(sa)
                perm_b.append(sb)
        perm_diff = sum(perm_a) / n - sum(perm_b) / n
        # 双尾检验
        if abs(perm_diff) >= abs(obs_diff):
            extreme_count += 1

    p_value = (extreme_count + 1) / (n_permutations + 1)

    return {
        "n_queries": n,
        "A_mean": round(mean_a, 6),
        "B_mean": round(mean_b, 6),
        "Diff": round(obs_diff, 6),
        "n_permutations": n_permutations,
        "p_value": round(p_value, 6),
        "alpha": 0.05,
        "significant": p_value < 0.05,
    }


# ============================================================
# IR Metrics (TREC-style)
# ============================================================

def load_qrels(path: str | Path) -> Dict[str, Dict[str, int]]:
    """加载 TREC 格式 qrels: qid doc_id relevance"""
    qrels: Dict[str, Dict[str, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
        qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def load_run(path: str | Path) -> Dict[str, List[str]]:
    """加载 JSON run 文件: {"qid": [doc_id1, doc_id2, ...]} 或 TREC 格式"""
    content = Path(path).read_text(encoding="utf-8").strip()
    # Try JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return {qid: docs for qid, docs in data.items() if isinstance(docs, list)}
    except json.JSONDecodeError:
        pass

    # TREC format: qid Q0 doc_id rank score run_name
    run: Dict[str, List[str]] = {}
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        qid = parts[0]
        doc_id = parts[2]
        run.setdefault(qid, []).append(doc_id)
    return run


def dcg_at_k(scores: List[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k(run_docs: List[str], qrel_docs: Dict[str, int], k: int) -> float:
    ideal = sorted(qrel_docs.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal[:k], k)
    if ideal_dcg == 0:
        return 0.0
    run_scores = [qrel_docs.get(d, 0) for d in run_docs[:k]]
    return dcg_at_k(run_scores, k) / ideal_dcg


def precision_at_k(run_docs: List[str], qrel_docs: Dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    relevant = sum(1 for d in run_docs[:k] if qrel_docs.get(d, 0) > 0)
    return relevant / k


def recall_at_k(run_docs: List[str], qrel_docs: Dict[str, int], k: int) -> float:
    total_relevant = sum(1 for v in qrel_docs.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    relevant = sum(1 for d in run_docs[:k] if qrel_docs.get(d, 0) > 0)
    return relevant / total_relevant


def mrr(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    reciprocals: List[float] = []
    for qid in run:
        if qid not in qrels:
            continue
        rel_docs = qrels[qid]
        for i, doc_id in enumerate(run[qid][:k], 1):
            if rel_docs.get(doc_id, 0) > 0:
                reciprocals.append(1.0 / i)
                break
        else:
            reciprocals.append(0.0)
    return sum(reciprocals) / len(reciprocals) if reciprocals else 0.0


def compute_ir_metrics(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """计算完整的 IR 指标"""
    if ks is None:
        ks = [1, 3, 5, 10, 20]

    ndcg_vals: List[float] = []
    prec_vals: List[float] = []
    recall_vals: List[float] = []

    for qid in run:
        if qid not in qrels:
            continue
        run_docs = run[qid]
        qrel_docs = qrels[qid]
        k = max(ks)
        ndcg_vals.append(ndcg_at_k(run_docs, qrel_docs, k))
        prec_vals.append(precision_at_k(run_docs, qrel_docs, k))
        recall_vals.append(recall_at_k(run_docs, qrel_docs, k))

    return {
        "mrr": mrr(run, qrels),
        "ndcg@k": sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else 0.0,
        "precision@k": sum(prec_vals) / len(prec_vals) if prec_vals else 0.0,
        "recall@k": sum(recall_vals) / len(recall_vals) if recall_vals else 0.0,
    }


# ============================================================
# CLI Entry Point
# ============================================================

def _cmd_compare(args: List[str]) -> int:
    """python -m app.rag.eval compare run_a.json run_b.json qrels.jsonl"""
    if len(args) < 2:
        print("Usage: python -m app.rag.eval compare run_a.json run_b.json [qrels.jsonl]")
        return 1

    run_a_path = args[0]
    run_b_path = args[1]
    qrels_path = args[2] if len(args) > 2 else None

    run_a = load_run(run_a_path)
    run_b = load_run(run_b_path)

    if qrels_path:
        qrels = load_qrels(qrels_path)
        metrics_a = compute_ir_metrics(run_a, qrels)
        metrics_b = compute_ir_metrics(run_b, qrels)
    else:
        metrics_a = {"mrr": 0.0, "ndcg@k": 0.0}
        metrics_b = {"mrr": 0.0, "ndcg@k": 0.0}

    # 按查询粒度收集分数用于 p-value
    common_qids = set(run_a.keys()) & set(run_b.keys())

    if qrels_path:
        qrels = load_qrels(qrels_path)
        scores_a = []
        scores_b = []
        for qid in sorted(common_qids):
            if qid not in qrels:
                continue
            scores_a.append(ndcg_at_k(run_a[qid], qrels[qid], 10))
            scores_b.append(ndcg_at_k(run_b[qid], qrels[qid], 10))

        result = permutation_test(scores_a, scores_b)
        print("\n=== p-value 检验 (双尾置换检验) ===")
        print(f"N queries   : {result['n_queries']}")
        print(f"A_mean      : {result['A_mean']:.6f}")
        print(f"B_mean      : {result['B_mean']:.6f}")
        print(f"Diff        : {result['Diff']:.6f}")
        print(f"p_value     : {result['p_value']:.6f}")
        print(f"significant : {result['significant']} (alpha={result['alpha']})")

        print("\n=== IR 指标对比 ===")
        print(f"{'Metric':<15} {'Sys A':>10} {'Sys B':>10}")
        for metric in sorted(metrics_a.keys()):
            print(f"{metric:<15} {metrics_a[metric]:>10.6f} {metrics_b[metric]:>10.6f}")
    else:
        print("No qrels provided, skipping p-value calculation")

    return 0


def _cmd_significance(args: List[str]) -> int:
    """python -m app.rag.eval significance scores_a.json scores_b.json"""
    if len(args) != 2:
        print("Usage: python -m app.rag.eval significance scores_a.json scores_b.json")
        return 1

    a_path, b_path = args[0], args[1]
    scores_a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    scores_b = json.loads(Path(b_path).read_text(encoding="utf-8"))

    result = permutation_test(scores_a, scores_b)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_ir_metrics(args: List[str]) -> int:
    """python -m app.rag.eval ir-metrics run.json qrels.jsonl"""
    if len(args) != 2:
        print("Usage: python -m app.rag.eval ir-metrics run.json qrels.jsonl")
        return 1

    run = load_run(args[0])
    qrels = load_qrels(args[1])
    metrics = compute_ir_metrics(run, qrels)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag.eval <subcommand> [args]")
        print("  compare          : A/B 系统对比 + p-value 检验")
        print("  significance      : 仅 p-value 检验")
        print("  ir-metrics        : 单个系统 IR 指标计算")
        sys.exit(1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    commands = {
        "compare": _cmd_compare,
        "significance": _cmd_significance,
        "ir-metrics": _cmd_ir_metrics,
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    sys.exit(commands[cmd](rest))


if __name__ == "__main__":
    main()
