"""Home, Upload/Index management and Repository overview pages."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis.repo_summary import analyze_repository
from src.ingestion.zip_loader import ZipValidationError, extract_repository
from src.rag.llm import llm_status
from src.retrieval.indexer import index_repository
from src.ui import state
from src.ui.theme import badge, note


def page_home() -> None:
    st.title("🧭 RepoPilot AI")
    st.caption("Codebase intelligence: upload a repo, search it semantically, ask grounded questions, scan for risks, and measure retrieval quality.")
    s = state.settings()
    repos = state.list_repos()
    repo = state.active_repo()
    ready, why = llm_status(s)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Repositories", len(repos))
    c2.metric("Active repo", repo["name"] if repo else "—")
    c3.metric("Indexed chunks", state.store().count(repo["id"]) if repo else 0, help="Chunks stored in ChromaDB for the active repo.")
    c4.metric("Embedding backend", s.embedding_backend)
    st.markdown(
        badge("LLM generation ON", "green") if ready else badge("Retrieval-only mode (no LLM)", "amber"),
        unsafe_allow_html=True,
    )
    if not ready:
        st.caption(why)

    st.subheader("Quick start")
    a, b = st.columns([1, 2])
    with a:
        if st.button("📦 Load bundled sample repository", help="Copies the small ShelfLife demo repo into your workspace.", width="stretch"):
            state.load_sample_repo()
            st.success("Sample repository loaded. Go to **Upload → Index management** to index it.")
            st.rerun()
    with b:
        st.markdown(
            "1. **Upload** a ZIP (or load the sample) and click *Index*.\n"
            "2. Browse the **Overview**, run **Semantic search**, or ask the **Chat**.\n"
            "3. Run **Static analysis**, then the **Evaluation** dashboard to see retrieval metrics."
        )
    note("Uploaded code is only read as text; it is never executed. Nothing leaves your machine unless you enable an LLM provider in Settings.")


def _index_controls(repo: dict) -> None:
    s = state.settings()
    st_ = state.store()
    count = st_.count(repo["id"])
    built_with = st_.embedder_name(repo["id"])
    cols = st.columns(4)
    cols[0].metric("Chunks indexed", count)
    cols[1].metric("Files", repo["files"])
    cols[2].metric("Index built with", built_with or "—")
    cols[3].metric("Skipped files", repo["skipped"])
    b1, b2, b3 = st.columns(3)
    label = "Re-index" if count else "Index"
    if b1.button(f"⚙️ {label}", type="primary", width="stretch", help="Chunk, embed locally and store in ChromaDB. Replaces any existing index."):
        emb = state.embedder()
        if emb is not None:
            bar = st.progress(0.0, text="Starting...")
            try:
                stats = index_repository(state.repo_dir(repo["id"]), repo["id"], st_, emb, s.chunk_max_lines,
                                         s.chunk_overlap_lines, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                st.success(f"Indexed {stats.chunks} chunks from {stats.files_indexed} files in {stats.seconds}s using {stats.embedder}.")
                for e in stats.errors[:5]:
                    st.warning(e)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Indexing failed: {type(exc).__name__}: {exc}")
            bar.empty()
    if b2.button("🗑️ Delete index", width="stretch", disabled=count == 0, help="Remove the vectors but keep the extracted files."):
        st_.delete(repo["id"])
        st.rerun()
    if b3.button("❌ Delete repository", width="stretch", help="Remove the index AND the extracted files."):
        state.delete_repo(repo["id"])
        st.rerun()


def page_upload() -> None:
    st.title("📤 Upload & index")
    s = state.settings()
    st.caption(f"Limits: ZIP ≤ {s.max_zip_mb} MB · ≤ {s.max_files} entries · ≤ {s.max_extracted_mb} MB extracted · files ≤ {s.max_file_kb} KB.")
    up = st.file_uploader("Repository ZIP", type=["zip"], help="Path traversal, symlinks, oversized and binary files are rejected or skipped.")
    if up is not None and st.button("Validate & extract", type="primary"):
        try:
            with st.spinner("Validating and extracting..."):
                rep = extract_repository(
                    up.getvalue(), s.repos_dir, max_zip_mb=s.max_zip_mb, max_files=s.max_files,
                    max_extracted_mb=s.max_extracted_mb, max_file_kb=s.max_file_kb,
                )
            state.register_repo(rep.repo_id, up.name.removesuffix(".zip"), len(rep.indexed_files), len(rep.skipped))
            st.session_state["repo_id"] = rep.repo_id
            st.success(f"Extracted {len(rep.indexed_files)} supported files ({len(rep.skipped)} skipped).")
            if rep.skipped:
                with st.expander("Skipped files and reasons"):
                    st.dataframe(pd.DataFrame(sorted(rep.skipped.items()), columns=["file", "reason"]), width="stretch", hide_index=True)
        except ZipValidationError as exc:
            st.error(f"Upload rejected: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error while extracting: {type(exc).__name__}: {exc}")

    st.subheader("Index management")
    repo = state.active_repo()
    if repo is None:
        st.info("No repository yet. Upload a ZIP above, or load the sample from **Home**.")
        return
    st.markdown(f"Active repository: **{repo['name']}** (`{repo['id']}`)")
    _index_controls(repo)


def page_overview() -> None:
    st.title("🗂️ Repository overview")
    repo = state.require_repo()
    if repo is None:
        return
    with st.spinner("Analysing repository (AST only, nothing is executed)..."):
        summ = analyze_repository(state.repo_dir(repo["id"]))
    c = st.columns(4)
    c[0].metric("Files", summ.total_files)
    c[1].metric("Lines of text", f"{summ.total_lines:,}")
    c[2].metric("Languages", len(summ.languages))
    c[3].metric("Python syntax errors", summ.python_stats.get("syntax_errors", 0))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Languages")
        df = pd.DataFrame([{"language": k, **v} for k, v in summ.languages.items()])
        if not df.empty:
            st.plotly_chart(px.bar(df, x="language", y="lines", hover_data=["files"], template="plotly_dark"), width="stretch")
    with right:
        st.subheader("File extensions")
        st.dataframe(pd.DataFrame(summ.extensions.items(), columns=["extension", "files"]), width="stretch", hide_index=True)

    st.subheader("Repository tree")
    st.code(summ.tree_text or "(empty)", language="text")

    st.subheader("Folders and apparent purpose")
    note("Purposes are <b>estimates</b> inferred from folder names, not verified by running or reading the code.")
    st.dataframe(pd.DataFrame(summ.folders), width="stretch", hide_index=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Dependencies")
        if summ.dependencies:
            for manifest, deps in summ.dependencies.items():
                with st.expander(f"{manifest} ({len(deps)})", expanded=True):
                    st.write(", ".join(deps))
        else:
            st.caption("No supported manifest (requirements.txt, package.json, pyproject.toml, pom.xml) found.")
    with b:
        st.subheader("Important config files")
        st.write("\n".join(f"- `{f}`" for f in summ.config_files) or "None detected.")
        st.subheader("Detected frameworks (estimate)")
        st.write(", ".join(summ.frameworks) or "None detected from Python imports.")

    if summ.routes:
        st.subheader("HTTP routes found in Python (AST)")
        st.dataframe(pd.DataFrame(summ.routes), width="stretch", hide_index=True)
    if summ.syntax_errors:
        st.subheader("Python files that failed to parse")
        st.dataframe(pd.DataFrame(summ.syntax_errors.items(), columns=["file", "error"]), width="stretch", hide_index=True)
