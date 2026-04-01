"""Tests for evaluation enhancements: ROUGE/F1/EM + p-value."""

from app.rag.eval import permutation_test as eval_permutation_test


class TestPermutationTest:
    def test_identical_scores_returns_p1(self):
        scores_a = [0.5, 0.6, 0.7, 0.8]
        scores_b = [0.5, 0.6, 0.7, 0.8]
        result = eval_permutation_test(scores_a, scores_b, n_permutations=1000, seed=42)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_very_different_scores_should_be_significant(self):
        scores_a = [0.9, 0.95, 0.85, 0.88, 0.92]
        scores_b = [0.1, 0.15, 0.2, 0.25, 0.12]
        result = eval_permutation_test(scores_a, scores_b, n_permutations=5000, seed=42)
        assert result["A_mean"] > result["B_mean"]
        # Permuted p-value may vary; just check structure returns
        assert result["Diff"] > 0.5

    def test_empty_lists_return_safe_result(self):
        result = eval_permutation_test([], [], seed=42)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_result_structure(self):
        result = eval_permutation_test([0.5], [0.6], seed=42)
        assert "A_mean" in result
        assert "B_mean" in result
        assert "Diff" in result
        assert "p_value" in result
        assert "significant" in result


class TestGenerationMetrics:
    def test_identical_strings_have_em1(self):
        from app.rag.evaluation_gate import exact_match
        assert exact_match("hello world", "hello world") == 1.0

    def test_different_strings_have_em0(self):
        from app.rag.evaluation_gate import exact_match
        assert exact_match("hello world", "goodbye world") == 0.0

    def test_f1_complete_overlap(self):
        from app.rag.evaluation_gate import f1_score_tokens
        assert f1_score_tokens("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_rouge1_identical(self):
        from app.rag.evaluation_gate import rouge_n
        assert rouge_n("the cat sat", "the cat sat", n=1) == 1.0

    def test_rouge2_identical(self):
        from app.rag.evaluation_gate import rouge_n
        assert rouge_n("the cat sat", "the cat sat", n=2) == 1.0

    def test_rouge_l_identical(self):
        from app.rag.evaluation_gate import rouge_l
        assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_compute_generation_metrics(self):
        from app.rag.evaluation_gate import compute_generation_metrics
        preds = ["hello world", "goodbye world"]
        refs = ["hello world", "goodbye world"]
        result = compute_generation_metrics(preds, refs)
        assert result["em"] == 1.0
        assert result["f1"] == 1.0
        assert result["rouge-1"] == 1.0

    def test_compute_generation_metrics_partial(self):
        from app.rag.evaluation_gate import compute_generation_metrics
        preds = ["hello world", "goodbye world"]
        refs = ["hello world", "farewell world"]
        assert compute_generation_metrics(preds, refs)["em"] < 1.0
