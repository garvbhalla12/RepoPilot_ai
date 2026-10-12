"""Deterministic grounding checks for generated answers (proxy checks, not proof)."""
from __future__ import annotations

import re
from typing import Sequence

from src.retrieval.store import Hit

_CITE = re.compile(r"\[(\d{1,2})\]")
_PATH = re.compile(r"[\w./\\-]+\.(?:py|js|jsx|ts|tsx|java|cpp|h|html|css|json|ya?ml|md|toml|txt|cfg|ini)\b")
INSUFFICIENT_MARKER = "insufficient evidence"


def check_grounding(answer: str, hits: Sequence[Hit]) -> dict:
    """Check an answer against the retrieved evidence.

    Flags: citation markers that point outside the retrieved set, and file paths
    mentioned in the answer that do not appear in any retrieved chunk. Passing these
    checks does NOT prove the answer is correct; it only catches obvious ungrounded references.
    """
    markers = [int(m) for m in _CITE.findall(answer)]
    valid = [m for m in markers if 1 <= m <= len(hits)]
    invalid = sorted(set(markers) - set(valid))
    retrieved_files = {h.file_path for h in hits}
    mentioned = {p.replace("\\", "/") for p in _PATH.findall(answer)}
    unsupported = sorted(
        p for p in mentioned if not any(rf == p or rf.endswith("/" + p) or p.endswith("/" + rf) for rf in retrieved_files)
    )
    declined = INSUFFICIENT_MARKER in answer.lower()
    return {
        "has_citations": bool(valid),
        "citation_markers": sorted(set(valid)),
        "invalid_citation_markers": invalid,
        "mentioned_files": sorted(mentioned),
        "unsupported_files": unsupported,
        "declined_for_insufficient_evidence": declined,
        "passes": declined or (bool(valid) and not invalid and not unsupported),
    }
