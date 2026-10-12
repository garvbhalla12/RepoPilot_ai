"""Local embedding backends.

* ``sentence-transformers`` (default): neural semantic embeddings, run locally.
  The model is downloaded once from the Hugging Face Hub on first use.
* ``hashing``: deterministic, dependency-free *lexical* embeddings (signed hashed
  bag-of-words over code identifiers). It needs no download, so tests and offline
  demos work, but it matches vocabulary rather than meaning. It is NOT a neural model.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol, Sequence


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the requested embedding backend cannot be loaded."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = {"the", "a", "an", "of", "to", "is", "in", "and", "or", "for", "on", "it", "this", "that", "with", "how", "what", "where", "does", "do", "are", "be", "by", "from", "as", "at"}


def tokenize_code(text: str) -> list[str]:
    """Lowercase tokens; splits snake_case and camelCase and keeps the joined identifier."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        parts = [p.lower() for p in _CAMEL.findall(word)]
        if len(parts) > 1:
            tokens.append(word.lower())
        tokens.extend(parts)
    return [t for t in tokens if t not in _STOP and len(t) > 1]


class HashingEmbedder:
    """Deterministic signed feature-hashing embedder (lexical, offline)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize_code(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        counts: dict[str, int] = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        for f, c in counts.items():
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            idx, sign = h % self.dim, 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(c))
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence_transformers.SentenceTransformer`` (local inference)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "sentence-transformers is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # network, disk, bad model name
            raise EmbeddingUnavailableError(
                f"Could not load embedding model '{model_name}': {exc}. "
                "The first run needs internet access to download it; "
                "alternatively set EMBEDDING_BACKEND=hashing for an offline lexical fallback."
            ) from exc
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vecs]


@lru_cache(maxsize=4)
def get_embedder(backend: str = "sentence-transformers", model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Return a cached embedder for the given backend."""
    backend = backend.strip().lower()
    if backend == "hashing":
        return HashingEmbedder()
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(model)
    raise EmbeddingUnavailableError(f"Unknown embedding backend: {backend!r}")
