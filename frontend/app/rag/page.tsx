"use client";

import Link from "next/link";
import { KeyboardEvent, useCallback, useEffect, useState } from "react";
import styles from "./page.module.css";

type RagPreset = "balanced" | "consult_strict" | "booking_fast";

interface SourceItem {
  source?: string;
  url?: string;
}

interface RagResult {
  id?: number | string;
  title?: string;
  source?: string;
  department?: string;
  score?: number;
  content?: string;
  sources?: SourceItem[];
}

interface RagSearchResponse {
  query: string;
  top_k: number;
  use_rerank: boolean;
  rerank_threshold: number;
  count: number;
  results: RagResult[];
  debug?: Record<string, unknown> | null;
}

interface RagStatusResponse {
  status: string;
  bm25_ready: boolean;
  reranker_loaded: boolean;
}

const PRESET_CONFIG: Record<RagPreset, { label: string; topK: number; useRerank: boolean; rerankThreshold: number }> = {
  balanced: { label: "平衡默认", topK: 3, useRerank: true, rerankThreshold: 0.15 },
  consult_strict: { label: "问诊严格", topK: 5, useRerank: true, rerankThreshold: 0.25 },
  booking_fast: { label: "挂号快速", topK: 2, useRerank: false, rerankThreshold: 0.15 }
};
const DEFAULT_RAG_API_BASE = "http://127.0.0.1:8001";
const ENV_RAG_API_BASE = process.env.NEXT_PUBLIC_BACKEND_BASE_URL?.replace(/\/$/, "") ?? "";

function resolveRuntimeApiBase(): string {
  if (ENV_RAG_API_BASE) {
    return ENV_RAG_API_BASE;
  }
  if (typeof window === "undefined") {
    return DEFAULT_RAG_API_BASE;
  }
  const { protocol, hostname } = window.location;
  return `${protocol}//${hostname}:8001`;
}

function scoreText(score: number | undefined): string {
  if (typeof score !== "number" || Number.isNaN(score)) {
    return "N/A";
  }
  return `${(score * 100).toFixed(1)}%`;
}

function normalizeSources(result: RagResult): SourceItem[] {
  if (Array.isArray(result.sources) && result.sources.length > 0) {
    return result.sources;
  }
  if (result.source) {
    return [{ source: result.source }];
  }
  return [{ source: "未知来源" }];
}

interface SearchSnapshot {
  elapsedMs: number;
  topScore: number;
  count: number;
}

function scoreValue(result: RagResult | undefined): number {
  if (!result || typeof result.score !== "number" || Number.isNaN(result.score)) {
    return 0;
  }
  return result.score;
}

function deltaText(current: number, previous: number, suffix = ""): string {
  const diff = current - previous;
  const sign = diff >= 0 ? "+" : "";
  return `${sign}${diff.toFixed(1)}${suffix}`;
}

export default function RagPage() {
  const [query, setQuery] = useState("高血压患者长期用药要注意什么");
  const [topK, setTopK] = useState(3);
  const [useRerank, setUseRerank] = useState(true);
  const [pureMode, setPureMode] = useState(true);
  const [rerankThreshold, setRerankThreshold] = useState(0.15);
  const [includeDebug, setIncludeDebug] = useState(false);
  const [enableIntentRouter, setEnableIntentRouter] = useState(false);
  const [enableHyde, setEnableHyde] = useState(false);
  const [preset, setPreset] = useState<RagPreset>("balanced");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<RagSearchResponse | null>(null);
  const [statusText, setStatusText] = useState("检查服务状态中...");
  const [serviceOnline, setServiceOnline] = useState(false);
  const [statusMeta, setStatusMeta] = useState("BM25: - | Reranker: -");
  const [apiBase, setApiBase] = useState(ENV_RAG_API_BASE || DEFAULT_RAG_API_BASE);
  const [latestSnapshot, setLatestSnapshot] = useState<SearchSnapshot | null>(null);
  const [prevSnapshot, setPrevSnapshot] = useState<SearchSnapshot | null>(null);

  useEffect(() => {
    if (ENV_RAG_API_BASE) {
      return;
    }
    const runtimeBase = resolveRuntimeApiBase();
    setApiBase((prev) => (prev === runtimeBase ? prev : runtimeBase));
  }, []);

  const applyPreset = (nextPreset: RagPreset) => {
    const next = PRESET_CONFIG[nextPreset];
    setPreset(nextPreset);
    setTopK(next.topK);
    setUseRerank(next.useRerank);
    setRerankThreshold(next.rerankThreshold);
  };

  const checkStatus = useCallback(async () => {
    try {
      const resp = await fetch(`${apiBase}/api/v1/rag/status`, { cache: "no-store" });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const payload = (await resp.json()) as RagStatusResponse;
      setServiceOnline(payload.status === "ok");
      setStatusText(payload.status === "ok" ? "服务运行中" : "服务异常");
      setStatusMeta(`BM25: ${payload.bm25_ready ? "READY" : "NOT_READY"} | Reranker: ${payload.reranker_loaded ? "LOADED" : "UNLOADED"}`);
    } catch {
      setServiceOnline(false);
      setStatusText("服务不可用");
      setStatusMeta("BM25: - | Reranker: -");
    }
  }, [apiBase]);

  useEffect(() => {
    void checkStatus();
    const timer = window.setInterval(() => void checkStatus(), 30000);
    return () => window.clearInterval(timer);
  }, [checkStatus]);

  const runSearch = async () => {
    const trimmed = query.trim();
    if (!trimmed) {
      setError("请输入检索问题。");
      return;
    }

    setPending(true);
    setError("");
    setResponse(null);
    const start = performance.now();
    try {
      const resp = await fetch(`${apiBase}/api/v1/rag/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        cache: "no-store",
        body: JSON.stringify({
          query: trimmed,
          top_k: topK,
          pure_mode: pureMode,
          use_rerank: useRerank,
          rerank_threshold: rerankThreshold,
          include_debug: includeDebug,
          enable_intent_router: enableIntentRouter,
          enable_hyde: enableHyde
        })
      });

      const payload = (await resp.json().catch(() => ({}))) as RagSearchResponse & { detail?: string };
      if (!resp.ok) {
        throw new Error(payload.detail || `检索失败: HTTP ${resp.status}`);
      }

      setResponse(payload);
      const elapsedMs = performance.now() - start;
      const nextSnapshot: SearchSnapshot = {
        elapsedMs,
        topScore: scoreValue(payload.results?.[0]),
        count: payload.count ?? payload.results?.length ?? 0
      };
      setPrevSnapshot(latestSnapshot);
      setLatestSnapshot(nextSnapshot);
      void checkStatus();
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      setError(`${reason}（top_k=${topK}, use_rerank=${useRerank}, rerank_threshold=${rerankThreshold}）`);
    } finally {
      setPending(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void runSearch();
    }
  };

  return (
    <main className={styles.pageRoot}>
      <div className={styles.bgPattern} />
      <div className={styles.gridOverlay} />

      <div className={styles.container}>
        <header className={styles.header}>
          <div className={styles.logo}>
            <div className={styles.logoIcon}>🔍</div>
            <div>
              <h1>RAG 医学文档检索</h1>
              <p className={styles.subtitle}>基于向量语义的智能医学知识检索系统</p>
            </div>
          </div>
          <Link href="/" className={styles.backBtn}>
            返回聊天页
          </Link>
        </header>

        <section className={styles.searchSection}>
          <div className={styles.searchForm}>
            <div className={styles.inputGroup}>
              <label htmlFor="rag-query">检索问题</label>
              <span className={styles.inputIcon}>🔎</span>
              <input
                id="rag-query"
                type="text"
                className={styles.searchInput}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="请输入您想检索的医学问题..."
                autoComplete="off"
              />
            </div>

            <div className={styles.topkGroup}>
              <label htmlFor="rag-topk">返回数量</label>
              <input
                id="rag-topk"
                type="number"
                className={styles.topkInput}
                min={1}
                max={10}
                value={topK}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  if (Number.isFinite(next)) {
                    setTopK(Math.max(1, Math.min(10, Math.floor(next))));
                  }
                }}
              />
            </div>

            <button
              type="button"
              className={`${styles.searchBtn} ${pending ? styles.loading : ""}`}
              disabled={pending}
              onClick={() => void runSearch()}
            >
              <span className={styles.btnText}>开始检索</span>
              <span className={styles.spinner} />
            </button>
          </div>

          <div className={styles.advancedRow}>
            <div className={styles.presetRow}>
              {(Object.keys(PRESET_CONFIG) as RagPreset[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  className={`${styles.presetBtn} ${preset === key ? styles.activePreset : ""}`}
                  onClick={() => applyPreset(key)}
                >
                  {PRESET_CONFIG[key].label}
                </button>
              ))}
            </div>

            <label className={styles.switchLabel}>
              <input type="checkbox" checked={pureMode} onChange={(event) => setPureMode(event.target.checked)} />
              pure_mode
            </label>
            <label className={styles.switchLabel}>
              <input type="checkbox" checked={useRerank} onChange={(event) => setUseRerank(event.target.checked)} />
              use_rerank
            </label>
            <label className={styles.switchLabel}>
              <input type="checkbox" checked={enableIntentRouter} onChange={(event) => setEnableIntentRouter(event.target.checked)} />
              intent_router
            </label>
            <label className={styles.switchLabel}>
              <input type="checkbox" checked={enableHyde} onChange={(event) => setEnableHyde(event.target.checked)} />
              hyde
            </label>
            <label className={styles.fieldLabel}>
              rerank_threshold
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                disabled={!useRerank}
                value={rerankThreshold}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  if (Number.isFinite(next)) {
                    setRerankThreshold(Math.max(0, Math.min(1, next)));
                  }
                }}
              />
            </label>
            <label className={styles.switchLabel}>
              <input type="checkbox" checked={includeDebug} onChange={(event) => setIncludeDebug(event.target.checked)} />
              include_debug
            </label>
          </div>
        </section>

        <div className={styles.statusBar}>
          <div className={styles.statusItem}>
            <span className={`${styles.statusDot} ${serviceOnline ? "" : styles.offline}`} />
            <span>{statusText}</span>
          </div>
          <div className={styles.statusItem}>
            <span>⚙️ {statusMeta}</span>
          </div>
          <div className={styles.statusItem}>
            <span>🧪 {pureMode ? "PURE" : "NORMAL"}</span>
          </div>
          <div className={styles.statusItem}>
            <span>🔗 {apiBase}</span>
          </div>
        </div>

        <section className={styles.resultsSection}>
          {latestSnapshot ? (
            <div className={styles.compareBar}>
              <div className={styles.compareItem}>
                <div className={styles.compareLabel}>本次耗时</div>
                <div className={styles.compareValue}>{latestSnapshot.elapsedMs.toFixed(1)} ms</div>
                {prevSnapshot ? <div className={styles.compareDelta}>{deltaText(latestSnapshot.elapsedMs, prevSnapshot.elapsedMs, " ms")}</div> : null}
              </div>
              <div className={styles.compareItem}>
                <div className={styles.compareLabel}>本次Top评分</div>
                <div className={styles.compareValue}>{(latestSnapshot.topScore * 100).toFixed(2)}%</div>
                {prevSnapshot ? <div className={styles.compareDelta}>{deltaText(latestSnapshot.topScore * 100, prevSnapshot.topScore * 100, "%")}</div> : null}
              </div>
              <div className={styles.compareItem}>
                <div className={styles.compareLabel}>结果数量</div>
                <div className={styles.compareValue}>{latestSnapshot.count}</div>
                {prevSnapshot ? <div className={styles.compareDelta}>{`${latestSnapshot.count - prevSnapshot.count >= 0 ? "+" : ""}${latestSnapshot.count - prevSnapshot.count}`}</div> : null}
              </div>
            </div>
          ) : null}

          <div className={styles.resultsHeader}>
            <h2 className={styles.resultsTitle}>检索结果</h2>
            <span className={styles.resultsCount}>{response ? `找到 ${response.count} 条结果` : "等待执行检索"}</span>
          </div>

          {error ? (
            <div className={styles.errorMessage}>
              <span className={styles.errorIcon}>⚠️</span>
              <span>{error}</span>
            </div>
          ) : null}

          <div className={styles.resultsContainer}>
            {!response?.results?.length ? (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>📖</div>
                <div className={styles.emptyText}>输入问题开始检索</div>
                <div className={styles.emptyHint}>支持中文医学问题的语义检索</div>
              </div>
            ) : (
              response.results.map((result, index) => {
                const sources = normalizeSources(result);
                return (
                  <article key={`${result.id ?? "doc"}-${index}`} className={styles.resultCard} style={{ animationDelay: `${index * 0.08}s` }}>
                    <div className={styles.resultHeader}>
                      <h3 className={styles.resultTitle}>{result.title || `检索结果 ${index + 1}`}</h3>
                      <div className={styles.resultScore}>
                        <span>⭐</span>
                        <span>{scoreText(result.score)}</span>
                      </div>
                    </div>
                    <div className={styles.resultContent}>{result.content || "无正文内容"}</div>
                    <div className={styles.resultSources}>
                      <div className={styles.sourcesLabel}>来源</div>
                      <div className={styles.sourceList}>
                        {sources.map((source, idx) => (
                          <div key={`src-${idx}`} className={styles.sourceItem}>
                            <span className={styles.sourceIcon}>📎</span>
                            <div className={styles.sourceText}>
                              <div>{source.source || "未知来源"}</div>
                              {source.url ? (
                                <a href={source.url} target="_blank" rel="noreferrer" className={styles.sourceUrl}>
                                  {source.url}
                                </a>
                              ) : null}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </article>
                );
              })
            )}
          </div>

          {response?.debug ? (
            <details className={styles.debugBlock}>
              <summary>调试指标</summary>
              <pre>{JSON.stringify(response.debug, null, 2)}</pre>
            </details>
          ) : null}
        </section>
      </div>
    </main>
  );
}
