import pytest

from src.evaluation import metrics as M


def test_precision_and_recall():
    retrieved = ["a.py", "b.py", "a.py", "c.py", "d.py"]
    assert M.precision_at_k(retrieved, ["a.py"], 5) == pytest.approx(0.4)
    assert M.precision_at_k(retrieved, ["a.py"], 1) == 1.0
    assert M.recall_at_k(retrieved, ["a.py", "z.py"], 5) == 0.5
    assert M.recall_at_k(retrieved, ["a.py", "b.py"], 2) == 1.0
    assert M.recall_at_k(retrieved, [], 5) == 0.0
    assert M.precision_at_k([], ["a.py"], 5) == 0.0


def test_reciprocal_rank_and_mrr():
    assert M.reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(["a"], ["a"]) == 1.0
    assert M.reciprocal_rank(["x"], ["a"]) == 0.0
    assert M.mean_reciprocal_rank([1.0, 0.5, 0.0]) == pytest.approx(0.5)
    assert M.mean_reciprocal_rank([]) == 0.0


def test_citation_file_correctness_uses_unique_files():
    assert M.citation_file_correctness(["a", "a", "b", "c"], ["a"], 4) == pytest.approx(1 / 3)
    assert M.citation_file_correctness([], ["a"], 3) == 0.0


def test_keyword_coverage_case_insensitive():
    assert M.keyword_coverage(["Uses PBKDF2 here"], ["pbkdf2", "sha256"]) == 0.5
    assert M.keyword_coverage(["x"], []) == 0.0


def test_percentile():
    assert M.percentile([1, 2, 3, 4, 5], 50) == 3
    assert M.percentile([10], 95) == 10
    assert M.percentile([], 95) == 0.0
