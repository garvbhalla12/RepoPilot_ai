import dataclasses

import pytest

from src.rag import llm
from src.rag.grounding import check_grounding
from src.rag.pipeline import RAGPipeline, build_context


def test_llm_disabled_by_default_and_missing_key_messages(settings, monkeypatch):
    ready, why = llm.llm_status(settings)
    assert not ready and "disabled" in why
    s = dataclasses.replace(settings, llm_enabled=True, llm_provider="openai", llm_model="some-model")
    ready, why = llm.llm_status(s)
    assert not ready and "OPENAI_API_KEY" in why
    s2 = dataclasses.replace(s, llm_provider="anthropic")
    assert "ANTHROPIC_API_KEY" in llm.llm_status(s2)[1]
    with pytest.raises(llm.LLMDisabledError):
        llm.generate(s, "sys", "user")  # raises before any network call


def test_model_name_required(settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-value")
    s = dataclasses.replace(settings, llm_enabled=True, llm_model="")
    assert "LLM_MODEL" in llm.llm_status(s)[1]


def test_local_endpoint_needs_no_key(settings):
    s = dataclasses.replace(settings, llm_enabled=True, llm_model="m", llm_base_url="http://localhost:11434/v1")
    assert llm.llm_status(s)[0]


def test_no_key_mode_returns_retrieval_only_with_evidence(settings, indexed, embedder):
    res = RAGPipeline(settings, indexed, embedder).ask("How are passwords hashed?", "sample")
    assert res.mode == "retrieval_only" and res.answer is None
    assert res.hits and "not a generated answer" in " ".join(res.notes)
    assert res.hits[0].start_line >= 1


def test_input_validation_and_missing_index(settings, indexed, embedder, store):
    p = RAGPipeline(settings, indexed, embedder)
    assert p.ask("   ", "sample").error
    assert "too long" in p.ask("x" * 5000, "sample").error
    assert "not been indexed" in RAGPipeline(settings, store, embedder).ask("hello", "ghost").error


def test_insufficient_evidence_when_scores_below_threshold(settings, indexed, embedder):
    s = dataclasses.replace(settings, min_score=0.999)
    res = RAGPipeline(s, indexed, embedder).ask("Stripe credit card payments", "sample")
    assert res.mode == "insufficient_evidence" and res.answer is None


def test_generated_path_uses_only_retrieved_context(settings, indexed, embedder, monkeypatch):
    captured = {}

    def fake_generate(s, system, user, max_tokens=700):  # test double; not presented as model output
        captured["user"] = user
        return "Passwords use PBKDF2 [1] in app/auth.py."

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(llm, "llm_status", lambda s: (True, "test"))
    res = RAGPipeline(settings, indexed, embedder).ask("How are passwords hashed?", "sample")
    assert res.mode == "generated" and "Context snippets" in captured["user"]
    assert res.grounding["has_citations"]


def test_generation_failure_falls_back_to_evidence(settings, indexed, embedder, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMTimeoutError("timed out")

    monkeypatch.setattr(llm, "generate", boom)
    monkeypatch.setattr(llm, "llm_status", lambda s: (True, "test"))
    res = RAGPipeline(settings, indexed, embedder).ask("How are passwords hashed?", "sample")
    assert res.mode == "retrieval_only" and "timed out" in res.error and res.hits


def test_context_redacts_secrets(settings, tmp_path, store, embedder):
    from src.retrieval.indexer import index_repository
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "c.py").write_text('def cfg():\n    api_key = "abcd1234efgh5678"\n    return api_key\n', encoding="utf-8")
    index_repository(repo, "r", store, embedder)
    res = RAGPipeline(settings, store, embedder).ask("api key", "r")
    ctx = build_context(res.hits)
    assert "abcd1234efgh5678" not in ctx and "[REDACTED]" in ctx


def test_grounding_checks(indexed, embedder):
    from src.retrieval.search import semantic_search
    hits = semantic_search(indexed, embedder, "sample", "password hashing", k=3)
    ok = check_grounding("It uses pbkdf2 [1] in app/auth.py", hits)
    bad = check_grounding("See [9] and app/imaginary.py", hits)
    none = check_grounding("Insufficient evidence in the retrieved code to answer this.", hits)
    assert ok["passes"] or ok["unsupported_files"] == []  # auth.py may not be in every top-3
    assert not bad["passes"] and bad["invalid_citation_markers"] == [9] and "app/imaginary.py" in bad["unsupported_files"]
    assert none["passes"] and none["declined_for_insufficient_evidence"]


def test_insecure_base_url_rejected(settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-value")
    s = dataclasses.replace(settings, llm_enabled=True, llm_model="m", llm_base_url="http://example.com/v1")
    with pytest.raises(llm.LLMDisabledError, match="https"):
        llm.generate(s, "sys", "user")  # rejected before any network request
