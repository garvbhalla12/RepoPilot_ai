"""Streamlit session state, cached resources and the repository registry."""
from __future__ import annotations

import dataclasses
import json
import shutil
import time
from pathlib import Path

import streamlit as st

from config import ROOT_DIR, Settings, get_settings
from src.embeddings.embedder import Embedder, EmbeddingUnavailableError, get_embedder
from src.retrieval.store import VectorStore

SAMPLE_REPO_ID = "sample-demo"


def settings() -> Settings:
    """Environment settings with any per-session overrides from the Settings page."""
    s = get_settings()
    overrides = st.session_state.get("overrides", {})
    return dataclasses.replace(s, **overrides) if overrides else s


def set_override(**kw) -> None:
    st.session_state.setdefault("overrides", {}).update(kw)
    st.session_state.pop("emb_fail", None)


@st.cache_resource(show_spinner=False)
def _store(path: str) -> VectorStore:
    return VectorStore(Path(path))


def store() -> VectorStore:
    s = settings()
    s.ensure_dirs()
    return _store(str(s.chroma_dir))


def reset_store_cache() -> None:
    _store.clear()


def embedder() -> Embedder | None:
    """Return the active embedder, or ``None`` after showing a helpful error."""
    s = settings()
    key = f"{s.embedding_backend}|{s.embedding_model}"
    fails = st.session_state.setdefault("emb_fail", {})
    if key in fails:
        st.error(fails[key])
        return None
    try:
        with st.spinner("Loading embedding backend (first run may download a model)..."):
            return get_embedder(s.embedding_backend, s.embedding_model)
    except EmbeddingUnavailableError as exc:
        fails[key] = str(exc)
        st.error(str(exc))
        return None


# ---- repository registry -------------------------------------------------------------
def _meta_path(s: Settings, rid: str) -> Path:
    return s.repos_dir / f"{rid}.json"


def repo_dir(rid: str) -> Path:
    return settings().repos_dir / rid


def register_repo(rid: str, name: str, files: int, skipped: int) -> None:
    s = settings()
    s.ensure_dirs()
    _meta_path(s, rid).write_text(json.dumps({
        "id": rid, "name": name, "files": files, "skipped": skipped, "added": time.strftime("%Y-%m-%d %H:%M"),
    }), encoding="utf-8")


def list_repos() -> list[dict]:
    s = settings()
    out = []
    if s.repos_dir.exists():
        for p in sorted(s.repos_dir.glob("*.json")):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                if (s.repos_dir / m["id"]).is_dir():
                    out.append(m)
            except (ValueError, KeyError, OSError):
                continue
    return out


def delete_repo(rid: str) -> None:
    s = settings()
    store().delete(rid)
    shutil.rmtree(s.repos_dir / rid, ignore_errors=True)
    _meta_path(s, rid).unlink(missing_ok=True)
    if st.session_state.get("repo_id") == rid:
        st.session_state.pop("repo_id", None)
    st.session_state.pop(f"chat_{rid}", None)


def load_sample_repo() -> None:
    """Copy the bundled sample repository into the registry."""
    s = settings()
    s.ensure_dirs()
    dest = s.repos_dir / SAMPLE_REPO_ID
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(ROOT_DIR / "data" / "sample_repo", dest)
    register_repo(SAMPLE_REPO_ID, "ShelfLife (bundled sample)", sum(1 for p in dest.rglob("*") if p.is_file()), 0)
    st.session_state["repo_id"] = SAMPLE_REPO_ID


def active_repo() -> dict | None:
    repos = {r["id"]: r for r in list_repos()}
    rid = st.session_state.get("repo_id")
    return repos.get(rid) if rid else None


def require_repo(need_index: bool = False) -> dict | None:
    """Show an empty-state message and return ``None`` when no (indexed) repo is selected."""
    repo = active_repo()
    if repo is None:
        st.info("No repository selected. Go to **Upload** to add a ZIP, or load the bundled sample from **Home**.")
        return None
    if need_index and store().count(repo["id"]) == 0:
        st.warning(f"**{repo['name']}** has no index yet. Open **Upload → Index management** and click *Index*.")
        return None
    return repo
