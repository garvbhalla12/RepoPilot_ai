"""Index a repository directory into the vector store."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.chunking.chunker import Chunk, chunk_file
from src.embeddings.embedder import Embedder
from src.ingestion.filters import language_for
from src.ingestion.zip_loader import iter_repo_files, read_text
from src.retrieval.store import VectorStore


@dataclass
class IndexStats:
    repo_id: str
    files_indexed: int = 0
    chunks: int = 0
    seconds: float = 0.0
    embedder: str = ""
    errors: list[str] = field(default_factory=list)


def build_chunks(root: Path, max_lines: int, overlap: int, errors: list[str] | None = None) -> tuple[list[Chunk], int]:
    """Chunk every supported file under ``root``. Returns ``(chunks, files_seen)``."""
    chunks: list[Chunk] = []
    files = iter_repo_files(root)
    for rel in files:
        lang = language_for(rel)
        if lang is None:
            continue
        try:
            chunks.extend(chunk_file(rel, read_text(root / rel), lang, max_lines, overlap))
        except Exception as exc:  # one bad file must not abort indexing
            if errors is not None:
                errors.append(f"{rel}: {exc}")
    return chunks, len(files)


def index_repository(
    root: Path,
    repo_id: str,
    store: VectorStore,
    embedder: Embedder,
    max_lines: int = 60,
    overlap: int = 10,
    progress: Callable[[float, str], None] | None = None,
) -> IndexStats:
    """Chunk, embed and persist a repository. Re-running replaces the existing index."""
    t0 = time.perf_counter()
    stats = IndexStats(repo_id=repo_id, embedder=embedder.name)
    if not root.exists():
        raise FileNotFoundError(f"Repository directory not found: {root}")
    if progress:
        progress(0.05, "Chunking source files...")
    chunks, n_files = build_chunks(root, max_lines, overlap, stats.errors)
    if not chunks:
        raise ValueError("No indexable content found (repository is empty or has only unsupported files).")
    embeddings: list[list[float]] = []
    batch = 64
    for i in range(0, len(chunks), batch):
        embeddings.extend(embedder.embed([c.text for c in chunks[i:i + batch]]))
        if progress:
            progress(0.1 + 0.8 * min(1.0, (i + batch) / len(chunks)), f"Embedding {min(i + batch, len(chunks))}/{len(chunks)} chunks")
    if progress:
        progress(0.95, "Writing to ChromaDB...")
    stats.chunks = store.build(repo_id, chunks, embeddings, embedder.name)
    stats.files_indexed = n_files
    stats.seconds = round(time.perf_counter() - t0, 3)
    if progress:
        progress(1.0, "Done")
    return stats
