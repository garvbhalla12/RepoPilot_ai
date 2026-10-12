import pytest

from src.embeddings.embedder import HashingEmbedder, tokenize_code
from src.retrieval.indexer import index_repository
from src.retrieval.search import EmbedderMismatchError, IndexMissingError, semantic_search

from tests.conftest import SAMPLE_REPO


def test_tokenizer_splits_identifiers():
    toks = tokenize_code("calculateLateFee late_fee_per_day")
    assert {"calculate", "late", "fee", "per", "day"} <= set(toks)


def test_hashing_embedder_is_deterministic_and_normalised(embedder):
    a, b = embedder.embed(["hash the password"]), embedder.embed(["hash the password"])
    assert a == b
    assert abs(sum(v * v for v in a[0]) - 1.0) < 1e-6


def test_index_counts_and_persistence(indexed, embedder, tmp_path):
    assert indexed.count("sample") > 10
    assert indexed.embedder_name("sample") == embedder.name
    assert "sample" in indexed.list_repos()


def test_retrieval_ranking_finds_expected_files(indexed, embedder):
    cases = {
        "How are passwords hashed?": "app/auth.py",
        "How is the late fee calculated?": "app/loans.py",
        "Which route logs a member in?": "app/main.py",
    }
    for question, expected in cases.items():
        hits = semantic_search(indexed, embedder, "sample", question, k=3)
        assert expected in [h.file_path for h in hits], question


def test_scores_sorted_descending(indexed, embedder):
    hits = semantic_search(indexed, embedder, "sample", "send overdue email reminders", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_citation_metadata_matches_source_lines(indexed, embedder):
    hits = semantic_search(indexed, embedder, "sample", "pbkdf2 password hash", k=3)
    assert hits
    for h in hits:
        src = (SAMPLE_REPO / h.file_path).read_text(encoding="utf-8").splitlines()
        assert 1 <= h.start_line <= h.end_line <= len(src)
        assert h.text == "\n".join(src[h.start_line - 1:h.end_line])
        assert h.citation == f"{h.file_path}:{h.start_line}-{h.end_line}"


def test_language_filter(indexed, embedder):
    hits = semantic_search(indexed, embedder, "sample", "borrow a book", k=5, language="JavaScript")
    assert hits and all(h.language == "JavaScript" for h in hits)


def test_reindex_replaces_and_delete_works(store, embedder):
    index_repository(SAMPLE_REPO, "r", store, embedder)
    n = store.count("r")
    index_repository(SAMPLE_REPO, "r", store, embedder)
    assert store.count("r") == n  # replaced, not duplicated
    assert store.delete("r") and not store.exists("r") and not store.delete("r")


def test_search_errors(store, embedder, indexed):
    with pytest.raises(IndexMissingError):
        semantic_search(store, embedder, "nope", "anything")
    assert semantic_search(indexed, embedder, "sample", "   ") == []
    with pytest.raises(EmbedderMismatchError):
        semantic_search(indexed, HashingEmbedder(dim=128), "sample", "x")


def test_empty_repository_cannot_be_indexed(tmp_path, store, embedder):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No indexable"):
        index_repository(tmp_path / "empty", "e", store, embedder)
    with pytest.raises(FileNotFoundError):
        index_repository(tmp_path / "missing", "m", store, embedder)


def test_malformed_files_do_not_break_indexing(tmp_path, store, embedder):
    repo = tmp_path / "bad"
    repo.mkdir()
    (repo / "broken.py").write_text("def x(:\n  ???\n" * 5, encoding="utf-8")
    (repo / "ok.py").write_text("def fine():\n    return 'valid code here'\n", encoding="utf-8")
    (repo / "latin.py").write_bytes("x = 'caf\xe9 valid'\n".encode("latin-1"))
    stats = index_repository(repo, "bad", store, embedder)
    assert stats.chunks >= 2
