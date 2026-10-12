"""ChromaDB persistent vector store (one collection per repository)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.chunking.chunker import Chunk


@dataclass
class Hit:
    """A retrieved chunk. ``score`` is cosine similarity (1 - cosine distance).

    It is a relative ranking signal, NOT a probability of correctness.
    """

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    kind: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def collection_name(repo_id: str) -> str:
    """Chroma-safe collection name for a repo id."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", repo_id).strip("-_") or "repo"
    return f"repo-{safe}"[:200]


class VectorStore:
    """Thin wrapper over a persistent ChromaDB client."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        )

    def _get(self, repo_id: str):
        return self._client.get_collection(collection_name(repo_id))

    def exists(self, repo_id: str) -> bool:
        return collection_name(repo_id) in {c if isinstance(c, str) else c.name for c in self._client.list_collections()}

    def count(self, repo_id: str) -> int:
        return self._get(repo_id).count() if self.exists(repo_id) else 0

    def embedder_name(self, repo_id: str) -> str | None:
        if not self.exists(repo_id):
            return None
        return (self._get(repo_id).metadata or {}).get("embedder")

    def list_repos(self) -> list[str]:
        names = [c if isinstance(c, str) else c.name for c in self._client.list_collections()]
        return sorted(n[len("repo-"):] for n in names if n.startswith("repo-"))

    def build(self, repo_id: str, chunks: list[Chunk], embeddings: list[list[float]], embedder_name: str) -> int:
        """(Re)create the collection and add all chunks. Returns the number stored."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.delete(repo_id)
        col = self._client.create_collection(
            collection_name(repo_id), metadata={"hnsw:space": "cosine", "embedder": embedder_name}
        )
        batch = 256
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            col.add(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[i:i + batch],
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
            )
        return col.count()

    def query(self, repo_id: str, embedding: list[float], k: int = 5, where: dict | None = None) -> list[Hit]:
        if not self.exists(repo_id):
            return []
        col = self._get(repo_id)
        n = col.count()
        if n == 0:
            return []
        res = col.query(
            query_embeddings=[embedding], n_results=min(max(k, 1), n), where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(
                chunk_id=meta["chunk_id"], file_path=meta["file_path"], start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]), language=meta["language"], kind=meta.get("kind", "block"),
                text=doc, score=round(1.0 - float(dist), 4),
            ))
        return hits

    def delete(self, repo_id: str) -> bool:
        if not self.exists(repo_id):
            return False
        self._client.delete_collection(collection_name(repo_id))
        return True

    def reset_all(self) -> None:
        """Delete every collection (documented reset option)."""
        for name in [c if isinstance(c, str) else c.name for c in self._client.list_collections()]:
            self._client.delete_collection(name)
