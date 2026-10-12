"""RepoPilot AI — Streamlit entry point. Run with: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="RepoPilot AI", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

from src.security.secrets import install_log_redaction  # noqa: E402
from src.ui import state  # noqa: E402
from src.ui.pages_ai import page_analysis, page_chat, page_evaluation, page_search, page_settings  # noqa: E402
from src.ui.pages_repo import page_home, page_overview, page_upload  # noqa: E402
from src.ui.theme import apply_theme  # noqa: E402

install_log_redaction()
apply_theme()

PAGES = {
    "🏠 Home": page_home,
    "📤 Upload": page_upload,
    "🗂️ Overview": page_overview,
    "🔎 Semantic search": page_search,
    "💬 Chat": page_chat,
    "🛡️ Static analysis": page_analysis,
    "📊 Evaluation": page_evaluation,
    "⚙️ Settings": page_settings,
}

with st.sidebar:
    st.markdown("## 🧭 RepoPilot AI")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")
    st.divider()
    repos = state.list_repos()
    if repos:
        ids = [r["id"] for r in repos]
        names = {r["id"]: r["name"] for r in repos}
        current = st.session_state.get("repo_id")
        idx = ids.index(current) if current in ids else 0
        st.session_state["repo_id"] = st.selectbox("Active repository", ids, index=idx, format_func=lambda i: names[i])
    else:
        st.caption("No repositories yet.")
    st.caption("Local-first · code is never executed")

PAGES[choice]()
