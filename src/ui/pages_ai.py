"""Semantic search, RAG chat, static analysis, evaluation and settings pages."""
from __future__ import annotations

import platform
import shutil
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis.dashboard import run_analysis
from src.analysis.static_tools import tool_available
from src.evaluation.runner import export_report, run_evaluation
from src.ingestion.filters import LANGUAGE_BY_EXT
from src.rag import llm
from src.rag.pipeline import RAGPipeline, RAGResult
from src.rag.suggestions import suggest_fixes
from src.retrieval.search import EmbedderMismatchError, IndexMissingError, semantic_search
from src.ui import state
from src.ui.theme import badge, note
from config import ROOT_DIR

LANG_FENCE = {"Python": "python", "JavaScript": "javascript", "TypeScript": "typescript", "Java": "java",
              "C++": "cpp", "HTML": "html", "CSS": "css", "JSON": "json", "YAML": "yaml", "Markdown": "markdown"}


def _show_hits(hits) -> None:
    for i, h in enumerate(hits, start=1):
        with st.expander(f"[{i}] {h.citation} · {h.language} · score {h.score:.3f}", expanded=i == 1):
            st.code(h.text, language=LANG_FENCE.get(h.language, "text"), line_numbers=True)
            st.caption(f"Lines {h.start_line}–{h.end_line} (code box numbering starts at 1). Score = cosine similarity, a relative ranking signal — not a probability.")


def page_search() -> None:
    st.title("🔎 Semantic code search")
    repo = state.require_repo(need_index=True)
    if repo is None:
        return
    s = state.settings()
    q = st.text_input("Describe what you are looking for", placeholder="e.g. where are passwords hashed?",
                      help="Natural-language or identifier queries. Runs locally; no API key needed.")
    c1, c2 = st.columns(2)
    k = c1.slider("Results (K)", 1, 20, s.top_k)
    lang = c2.selectbox("Language filter", ["Any"] + sorted(set(LANGUAGE_BY_EXT.values())))
    if not q:
        st.caption("Enter a query to search the indexed chunks.")
        return
    emb = state.embedder()
    if emb is None:
        return
    try:
        t0 = time.perf_counter()
        hits = semantic_search(state.store(), emb, repo["id"], q, k=k, language=None if lang == "Any" else lang)
        ms = (time.perf_counter() - t0) * 1000
    except (IndexMissingError, EmbedderMismatchError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Search failed: {type(exc).__name__}: {exc}")
        return
    st.caption(f"{len(hits)} results in {ms:.0f} ms using `{emb.name}`")
    if not hits:
        st.info("No results. Try different wording or remove the language filter.")
    _show_hits(hits)


def _render_result(res: RAGResult) -> None:
    if res.error and not res.hits:
        st.error(res.error)
        return
    if res.mode == "generated":
        st.markdown(badge("Generated answer", "green") + badge("verify against sources", "amber"), unsafe_allow_html=True)
        st.markdown(res.answer)
        if res.grounding:
            g = res.grounding
            st.caption(f"Automated grounding check: {'passed' if g['passes'] else 'FLAGGED'} · citations used: {g['citation_markers'] or 'none'}"
                       + (f" · unknown files: {g['unsupported_files']}" if g["unsupported_files"] else ""))
    elif res.mode == "insufficient_evidence":
        st.markdown(badge("Insufficient evidence", "red"), unsafe_allow_html=True)
        st.write("I could not find code in this repository that supports an answer, so I won't guess.")
    else:
        st.markdown(badge("Retrieval only — not a generated answer", "amber"), unsafe_allow_html=True)
    for n in res.notes:
        st.caption(n)
    if res.error:
        st.warning(res.error)
    if res.hits:
        with st.expander(f"Sources ({len(res.hits)}) · retrieval {res.retrieval_ms:.0f} ms"
                         + (f" · generation {res.generation_ms:.0f} ms" if res.generation_ms else ""), expanded=res.mode != "generated"):
            _show_hits(res.hits)


def page_chat() -> None:
    st.title("💬 Codebase chat")
    repo = state.require_repo(need_index=True)
    if repo is None:
        return
    s = state.settings()
    ready, why = llm.llm_status(s)
    if ready:
        st.markdown(badge("LLM generation enabled", "green"), unsafe_allow_html=True)
        st.caption(why + (" Retrieved snippets are sent to this provider." if not llm.is_local_endpoint(s) else ""))
        use_llm = st.toggle("Generate answers with the LLM", value=True)
    else:
        use_llm = False
        note("<b>No model provider enabled.</b> Grounded natural-language answers need a provider (see <b>Settings</b>). "
             "Until then, chat shows the retrieved evidence only — that is search, not a generated answer. " + why)
    key = f"chat_{repo['id']}"
    history: list[dict] = st.session_state.setdefault(key, [])
    if st.button("Clear conversation", disabled=not history):
        st.session_state[key] = []
        st.rerun()
    for m in history:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["content"])
            else:
                _render_result(m["result"])
    q = st.chat_input("Ask about architecture, auth, routes, database config, dependencies...")
    if q:
        history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        emb = state.embedder()
        if emb is None:
            return
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence..." if not use_llm else "Retrieving and generating..."):
                res = RAGPipeline(s, state.store(), emb).ask(q, repo["id"], use_llm=use_llm)
            _render_result(res)
        history.append({"role": "assistant", "result": res})


def page_analysis() -> None:
    st.title("🛡️ Static analysis & risk dashboard")
    repo = state.require_repo()
    if repo is None:
        return
    note("<b>Not a security audit.</b> Findings come from Ruff, Bandit and simple rules. Heuristic secret detection has both false positives and false negatives. "
         "Secret values are always masked.")
    c1, c2 = st.columns(2)
    use_ruff = c1.checkbox("Ruff", value=True, disabled=not tool_available("ruff"), help="Lint (Python). Installed: " + str(tool_available("ruff")))
    use_bandit = c2.checkbox("Bandit", value=True, disabled=not tool_available("bandit"), help="Python security linter. Installed: " + str(tool_available("bandit")))
    if st.button("Run analysis", type="primary"):
        with st.spinner("Scanning (code is parsed, never executed)..."):
            st.session_state[f"analysis_{repo['id']}"] = run_analysis(state.repo_dir(repo["id"]), use_ruff, use_bandit)
    rep = st.session_state.get(f"analysis_{repo['id']}")
    if rep is None:
        st.caption("Click *Run analysis* to scan the active repository.")
        return
    for t in rep.tools:
        if not t.available or t.error:
            st.warning(f"{t.tool}: {t.error}")
    counts = rep.severity_counts()
    m = st.columns(4)
    m[0].metric("High", counts["high"])
    m[1].metric("Medium", counts["medium"])
    m[2].metric("Low", counts["low"])
    m[3].metric("Files affected", len({f["file"] for f in rep.scanner_findings + rep.heuristic_findings}))
    allf = rep.scanner_findings + rep.heuristic_findings
    if allf:
        df = pd.DataFrame(allf)
        fig = px.histogram(df, x="severity", color="source", category_orders={"severity": ["high", "medium", "low"]}, template="plotly_dark")
        st.plotly_chart(fig, width="stretch")
    t1, t2, t3 = st.tabs(["Scanner findings (Ruff / Bandit)", "RepoPilot heuristics (secrets & patterns)", "AI-generated suggestions"])
    with t1:
        if rep.scanner_findings:
            st.dataframe(pd.DataFrame(rep.scanner_findings), width="stretch", hide_index=True)
        else:
            st.caption("No scanner findings (or scanners not run).")
    with t2:
        if rep.heuristic_findings:
            st.dataframe(pd.DataFrame(rep.heuristic_findings), width="stretch", hide_index=True)
            st.download_button("Download masked report (CSV)", pd.DataFrame(rep.heuristic_findings).to_csv(index=False), "heuristic_findings_masked.csv", "text/csv")
        else:
            st.caption("No heuristic findings.")
    with t3:
        s = state.settings()
        ready, why = llm.llm_status(s)
        st.markdown(badge("AI-generated — not scanner output", "amber"), unsafe_allow_html=True)
        if not ready:
            st.info("Requires an enabled LLM provider. " + why)
        elif st.button("Generate remediation suggestions"):
            try:
                with st.spinner("Asking the model (only rule ids, messages and locations are sent)..."):
                    st.markdown(suggest_fixes(s, sorted(allf, key=lambda f: {"high": 0, "medium": 1}.get(f["severity"], 2))))
            except llm.LLMError as exc:
                st.error(str(exc))


def page_evaluation() -> None:
    st.title("📊 Evaluation dashboard")
    s = state.settings()
    note(
        "Metrics below are <b>automated proxies</b> computed against labels in <code>data/evaluation/questions.json</code> on the bundled sample repo. "
        "They are not human-verified answer quality, and no scores are shown until you run the evaluation."
    )
    ready, why = llm.llm_status(s)
    c1, c2 = st.columns(2)
    k = c1.slider("K", 1, 10, s.top_k, help="Number of chunks retrieved per question.")
    with_llm = c2.checkbox("Also generate answers (needs LLM)", value=False, disabled=not ready, help=why)
    if st.button("▶ Run evaluation", type="primary"):
        emb = state.embedder()
        if emb is not None:
            with st.spinner("Indexing the sample repo and running questions..."):
                rep = run_evaluation(s, emb, state.store(), ROOT_DIR / "data" / "sample_repo",
                                     ROOT_DIR / "data" / "evaluation" / "questions.json", k=k, with_llm=with_llm)
                export_report(rep, s.reports_dir, "evaluation_latest")
            st.session_state["eval_report"] = rep
    rep = st.session_state.get("eval_report")
    if rep is None:
        st.caption("Run the evaluation to see results. CLI equivalent: `python scripts/run_evaluation.py`.")
        return
    st.caption(f"Embedder `{rep.meta['embedder']}` · K={rep.meta['k']} · LLM used: {rep.meta['llm_used']} · {rep.meta['created_utc']}")
    sm = rep.summary
    keys = [kk for kk in sm if kk.startswith("precision") or kk.startswith("recall")] + ["mrr", "citation_file_correctness", "keyword_coverage"]
    cols = st.columns(len(keys))
    for col, kk in zip(cols, keys):
        col.metric(kk, "n/a" if sm[kk] is None else f"{sm[kk]:.3f}")
    d = st.columns(4)
    d[0].metric("Top-1 file accuracy", sm["top1_file_accuracy"])
    d[1].metric("Mean latency (ms)", sm["latency_ms_mean"])
    d[2].metric("p95 latency (ms)", sm["latency_ms_p95"])
    d[3].metric("Failed queries", sm["failed_queries"])
    if sm["abstention_rate_on_unanswerable"] is not None:
        st.caption(f"Abstention on unanswerable questions (top score below MIN_SCORE={s.min_score}): {sm['abstention_rate_on_unanswerable']}. "
                   "A low value means the similarity threshold alone does not reliably detect unanswerable questions.")
    df = pd.DataFrame(rep.rows)
    ans = df[df["answerable"]]
    if not ans.empty and "precision_at_k" in ans:
        long = ans.melt(id_vars="id", value_vars=["precision_at_k", "recall_at_k", "reciprocal_rank", "keyword_coverage"])
        st.plotly_chart(px.bar(long, x="id", y="value", color="variable", barmode="group", template="plotly_dark"), width="stretch")
    st.plotly_chart(px.bar(df, x="id", y="latency_ms", template="plotly_dark", title="Retrieval latency per question (ms)"), width="stretch")
    st.subheader("Per-question results")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "evaluation.csv", "text/csv")
    import json
    st.download_button("Download JSON", json.dumps({"meta": rep.meta, "summary": rep.summary, "rows": rep.rows}, indent=2), "evaluation.json", "application/json")


def page_settings() -> None:
    st.title("⚙️ Settings")
    s = state.settings()
    st.caption("Changes here apply to this browser session. For permanent changes edit `.env` (see `.env.example`).")
    st.subheader("Embeddings & retrieval")
    backend = st.selectbox("Embedding backend", ["sentence-transformers", "hashing"],
                           index=0 if s.embedding_backend.startswith("sentence") else 1,
                           help="'sentence-transformers' = local neural embeddings (downloads a model on first use). "
                                "'hashing' = offline lexical fallback, matches words not meaning.")
    if backend == "hashing":
        st.warning("Hashing embeddings are a lexical baseline, not semantic. Use them for offline demos and tests.")
    c = st.columns(4)
    chunk = c[0].number_input("Chunk max lines", 10, 400, s.chunk_max_lines)
    ov = c[1].number_input("Chunk overlap lines", 0, 100, s.chunk_overlap_lines)
    topk = c[2].number_input("Top-K", 1, 20, s.top_k)
    ms = c[3].number_input("Min similarity", 0.0, 1.0, float(s.min_score), 0.01, help="Below this the chat answers 'insufficient evidence'. Untuned default; adjust per backend.")
    if st.button("Apply retrieval settings"):
        state.set_override(embedding_backend=backend, chunk_max_lines=int(chunk), chunk_overlap_lines=int(ov), top_k=int(topk), min_score=float(ms))
        st.success("Applied. Re-index repositories built with a different backend or chunk size.")

    st.subheader("LLM provider (optional)")
    ready, why = llm.llm_status(s)
    st.markdown(badge("enabled", "green") if ready else badge("disabled", "amber"), unsafe_allow_html=True)
    st.caption(why)
    enable = st.checkbox("Enable LLM generation", value=s.llm_enabled,
                         help="When enabled, retrieved snippets (secret-like values redacted) are sent to the provider configured in .env.")
    if enable:
        st.warning("With an external provider, retrieved code snippets leave your machine. Only enable this for code you are allowed to share.")
    if st.button("Apply LLM setting"):
        state.set_override(llm_enabled=enable)
        st.rerun()
    st.caption("Provider, model and API key are read from environment variables / `.env` and are never stored by the app.")

    st.subheader("Storage & reset")
    st.code(str(s.data_dir), language="text")
    st.caption("Contains uploaded repos, the Chroma index and evaluation reports. It is git-ignored.")
    confirm = st.checkbox("I understand this deletes ALL repositories, indexes and reports")
    if st.button("🧨 Reset all local data", disabled=not confirm):
        state.reset_store_cache()
        shutil.rmtree(s.data_dir, ignore_errors=True)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("All local data deleted.")
        st.rerun()

    with st.expander("Environment"):
        st.write({"python": platform.python_version(), "ruff": tool_available("ruff"), "bandit": tool_available("bandit"),
                  "embedding_model": s.embedding_model})
