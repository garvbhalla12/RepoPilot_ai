"""Semantic search over an indexed repository."""
from __future__ import annotations

from src.embeddings.embedder import Embedder
from src.retrieval.store import Hit, VectorStore


class IndexMissingError(RuntimeError):
    """Raised when a search is attempted before the repository is indexed."""


class EmbedderMismatchError(RuntimeError):
    """Raised when the query embedder differs from the one used to build the index."""


def semantic_search(
    store: VectorStore,
    embedder: Embedder,
    repo_id: str,
    query: str,
    k: int = 5,
    min_score: float | None = None,
    language: str | None = None,
) -> list[Hit]:
    """Return the top-``k`` chunks for ``query``, optionally filtered by language and score."""
    query = (query or "").strip()
    if not query:
        return []
    if not store.exists(repo_id) or store.count(repo_id) == 0:
        raise IndexMissingError("This repository has not been indexed yet. Index it first.")
    built_with = store.embedder_name(repo_id)
    if built_with and built_with != embedder.name:
        raise EmbedderMismatchError(
            f"Index was built with '{built_with}' but the active embedder is '{embedder.name}'. Re-index to continue."
        )
    where = {"language": language} if language else None
    hits = store.query(repo_id, embedder.embed([query])[0], k=k, where=where)
    if min_score is not None:
        hits = [h for h in hits if h.score >= min_score]
    return hits
