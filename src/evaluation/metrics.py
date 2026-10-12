"""Retrieval metrics. All functions are pure and deterministic.

``retrieved`` is the ranked list of file paths, one entry per retrieved chunk
(duplicates preserved because several chunks may come from one file).
"""
from __future__ import annotations

from typing import Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks that come from a relevant file."""
    rel = set(relevant)
    top = list(retrieved)[:k]
    if k <= 0 or not top:
        return 0.0
    return sum(1 for f in top if f in rel) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant files that appear at least once in the top-k chunks."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(list(retrieved)[:k])) / len(rel)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1/rank of the first relevant chunk, or 0.0 if none is retrieved."""
    rel = set(relevant)
    for i, f in enumerate(retrieved, start=1):
        if f in rel:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rr_values: Sequence[float]) -> float:
    """Mean of per-query reciprocal ranks."""
    return sum(rr_values) / len(rr_values) if rr_values else 0.0


def citation_file_correctness(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Of the unique files that would be cited (top-k), the fraction that are relevant."""
    rel = set(relevant)
    cited = list(dict.fromkeys(list(retrieved)[:k]))
    return sum(1 for f in cited if f in rel) / len(cited) if cited else 0.0


def keyword_coverage(texts: Sequence[str], keywords: Sequence[str]) -> float:
    """Fraction of expected keywords found (case-insensitive) in the retrieved text.

    A deterministic *evidence-support proxy*: it shows whether the retrieved context
    contains the facts a correct answer needs. It does not measure answer quality.
    """
    if not keywords:
        return 0.0
    blob = "\n".join(texts).lower()
    return sum(1 for kw in keywords if kw.lower() in blob) / len(keywords)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100])."""
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
