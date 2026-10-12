# Verification checklist

Everything below reflects what was actually run while building this project (Linux sandbox, Python 3.12.3).

## Executed and passing
- [x] `python -m pytest` → **64 passed** (ingestion/path traversal, filtering, chunking + line metadata, indexing, retrieval ranking,
      citation metadata, metrics, missing-key behaviour, malformed files, secrets masking, Ruff/Bandit integration, evaluation export, headless UI smoke tests).
      Tests use the offline `hashing` embedder.
- [x] `python scripts/run_evaluation.py --backend hashing --k 5` ran with no API key and exported JSON + CSV.
- [x] `streamlit run app.py` started headless; `/_stcore/health` and `/` returned HTTP 200.
- [x] Streamlit `AppTest` rendered all 8 pages and ran the sample workflow (load → index → search → chat → analysis → evaluation).
- [x] Ruff (E,F,W) clean on the project's own code; Bandit findings on own code reviewed (remaining are annotated, fixed-argv subprocess).

Measured once with the **hashing (lexical) backend**, K=5, on the 15-question bundled set (your numbers may differ with other backends):
precision@5 0.314 · recall@5 0.821 · MRR 0.643 · citation-file correctness 0.255 · keyword coverage 0.714 · top-1 file accuracy 0.5 ·
abstention on the unanswerable question 0.0 (the similarity threshold did NOT detect it). These are proxy metrics against author labels, not answer-quality scores.

## NOT tested (could not be done in the build sandbox)
- [ ] **`sentence-transformers` backend** — the install failed (disk space for torch); the code path (`SentenceTransformerEmbedder`) and its
      first-run model download were never executed. Expect to verify it on your machine; retrieval quality and the `MIN_SCORE` default are untuned for it.
- [ ] **Any real LLM provider call** (OpenAI-compatible / Anthropic). Generation, grounding and error handling are covered only with test doubles;
      HTTP request shapes were written from provider docs from memory and not exercised against a live API.
- [ ] **Windows PowerShell execution** — commands are standard but were run on Linux only.
- [ ] **Docker build/compose** — files provided, never built.
- [ ] **Visual appearance in a real browser** — only headless tests; no screenshots were produced.
- [ ] Python 3.11 specifically (tests ran on 3.12; `requirements.txt` pins are the versions installed there).

## Known limitations
- Hashing backend is lexical, not semantic; scores are low and noisy. Default `MIN_SCORE=0.05` is untuned.
- Evaluation set is small (14 answerable + 1 unanswerable) over a tiny repo; labels are author-written.
- Non-Python languages get only heuristic chunking/risk rules (no AST, no Ruff/Bandit).
- Large repositories are indexed synchronously in the UI thread.
