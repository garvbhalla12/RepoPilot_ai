"""Code-aware chunking that preserves file path and line-number metadata."""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a file. Line numbers are 1-based and inclusive."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    text: str
    kind: str = "block"  # function | class | module | block | doc

    def metadata(self) -> dict:
        """Metadata without the text (suitable for the vector store)."""
        d = asdict(self)
        d.pop("text")
        return d


_JS_BOUNDARY = re.compile(
    r"^\s*(export\s+)?(default\s+)?(async\s+)?(function\b|class\b|interface\b)"
    r"|^\s*(export\s+)?const\s+\w+\s*=\s*(async\s*)?(\(|function)"
    r"|^\s*(public|private|protected)\s+[\w<>\[\], ]+\s+\w+\s*\(.*\)\s*(throws [\w, ]+)?\s*\{?\s*$"
)
_MD_HEADING = re.compile(r"^#{1,6}\s+")


def _window(start: int, end: int, max_lines: int, overlap: int) -> list[tuple[int, int]]:
    """Split [start, end] (1-based inclusive) into windows of ``max_lines`` with overlap."""
    if end - start + 1 <= max_lines:
        return [(start, end)]
    step = max(1, max_lines - max(0, min(overlap, max_lines - 1)))
    spans: list[tuple[int, int]] = []
    s = start
    while s <= end:
        e = min(end, s + max_lines - 1)
        spans.append((s, e))
        if e == end:
            break
        s += step
    return spans


def _python_segments(source: str, total: int) -> list[tuple[int, int, str]] | None:
    """Top-level function/class segments via the AST; ``None`` if the file does not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    segs: list[tuple[int, int, str]] = []
    cursor = 1
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = min([n.lineno] + [d.lineno for d in n.decorator_list])
        end = n.end_lineno or n.lineno
        if start > cursor:
            segs.append((cursor, start - 1, "module"))
        segs.append((start, end, "class" if isinstance(n, ast.ClassDef) else "function"))
        cursor = end + 1
    if cursor <= total:
        segs.append((cursor, total, "module"))
    return segs


def _boundary_segments(lines: list[str], language: str) -> list[tuple[int, int, str]]:
    """Heuristic boundaries for JS/TS/Java (regex) and Markdown (headings)."""
    if language == "Markdown":
        matcher, kind = _MD_HEADING, "doc"
    elif language in {"JavaScript", "TypeScript", "Java"}:
        matcher, kind = _JS_BOUNDARY, "block"
    else:
        return [(1, len(lines), "block")]
    starts = [i + 1 for i, line in enumerate(lines) if matcher.match(line)]
    if not starts:
        return [(1, len(lines), "block")]
    if starts[0] != 1:
        starts.insert(0, 1)
    segs = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - 1 if i + 1 < len(starts) else len(lines)
        if e >= s:
            segs.append((s, e, kind))
    return segs


def chunk_file(
    file_path: str,
    text: str,
    language: str,
    max_lines: int = 60,
    overlap_lines: int = 10,
    min_chars: int = 20,
) -> list[Chunk]:
    """Chunk one file.

    Python uses the AST (function/class boundaries); JS/TS/Java/Markdown use
    boundary heuristics; everything else, and any oversized segment, falls back
    to overlapping line windows. Syntax errors never raise.
    """
    if max_lines < 1:
        raise ValueError("max_lines must be >= 1")
    lines = text.splitlines()
    if not lines or not text.strip():
        return []
    segs = _python_segments(text, len(lines)) if language == "Python" else None
    if segs is None:
        segs = _boundary_segments(lines, language)

    chunks: list[Chunk] = []
    for s, e, kind in segs:
        for ws, we in _window(s, e, max_lines, overlap_lines):
            body = "\n".join(lines[ws - 1:we])
            if len(body.strip()) < min_chars:
                continue
            chunks.append(Chunk(
                chunk_id=f"{file_path}:{ws}-{we}", file_path=file_path, start_line=ws,
                end_line=we, language=language, text=body, kind=kind,
            ))
    return chunks
