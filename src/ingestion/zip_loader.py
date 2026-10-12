"""Safe ZIP extraction. Uploaded code is only ever read as text, never executed."""
from __future__ import annotations

import io
import shutil
import stat
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from src.ingestion.filters import is_ignored_path, language_for, looks_binary, should_index


class ZipValidationError(ValueError):
    """Raised when an archive is unsafe or exceeds configured limits."""


@dataclass
class ExtractionReport:
    """Result of extracting a repository archive."""

    repo_id: str
    root: Path
    indexed_files: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    total_bytes: int = 0


def _is_unsafe_name(name: str) -> bool:
    norm = name.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        return True  # absolute POSIX path or Windows drive path
    return ".." in PurePosixPath(norm).parts or "\x00" in name


def validate_zip(zf: zipfile.ZipFile, *, max_files: int, max_extracted_mb: int) -> list[zipfile.ZipInfo]:
    """Validate archive entries and return the file (non-directory) entries.

    Rejects path traversal, absolute paths, symlinks, too many entries and
    oversized total uncompressed size (zip-bomb defence).
    """
    infos = zf.infolist()
    if len(infos) > max_files:
        raise ZipValidationError(f"Archive has {len(infos)} entries; the limit is {max_files}.")
    total = 0
    files: list[zipfile.ZipInfo] = []
    for info in infos:
        if _is_unsafe_name(info.filename):
            raise ZipValidationError(f"Unsafe path in archive: {info.filename!r}")
        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            raise ZipValidationError(f"Symlinks are not allowed: {info.filename!r}")
        if info.is_dir():
            continue
        total += info.file_size
        if total > max_extracted_mb * 1024 * 1024:
            raise ZipValidationError(f"Extracted size exceeds the {max_extracted_mb} MB limit.")
        files.append(info)
    if not files:
        raise ZipValidationError("Archive contains no files.")
    return files


def _common_root(names: list[str]) -> str:
    """If every path shares one top-level folder (typical GitHub ZIP), return it."""
    split = [PurePosixPath(n.replace("\\", "/")).parts for n in names]
    tops = {p[0] for p in split if p}
    if len(tops) == 1 and all(len(p) > 1 for p in split):
        return next(iter(tops))
    return ""


def extract_repository(
    source: BinaryIO | bytes,
    dest_root: Path,
    *,
    max_zip_mb: int = 50,
    max_files: int = 5000,
    max_extracted_mb: int = 200,
    max_file_kb: int = 512,
    repo_id: str | None = None,
) -> ExtractionReport:
    """Validate a ZIP and extract only indexable text files into ``dest_root/<repo_id>``."""
    data = source if isinstance(source, bytes) else source.read()
    if len(data) > max_zip_mb * 1024 * 1024:
        raise ZipValidationError(f"ZIP is larger than {max_zip_mb} MB.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ZipValidationError("File is not a valid ZIP archive.") from exc

    with zf:
        infos = validate_zip(zf, max_files=max_files, max_extracted_mb=max_extracted_mb)
        prefix = _common_root([i.filename for i in infos])
        repo_id = repo_id or uuid.uuid4().hex[:10]
        root = (dest_root / repo_id).resolve()
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        report = ExtractionReport(repo_id=repo_id, root=root)

        for info in infos:
            name = info.filename.replace("\\", "/")
            rel = name[len(prefix) + 1:] if prefix and name.startswith(prefix + "/") else name
            if not rel:
                continue
            with zf.open(info) as fh:
                head = fh.read(4096)
            ok, reason = should_index(rel, info.file_size, max_file_kb, head)
            if not ok:
                report.skipped[rel] = reason
                continue
            target = (root / rel).resolve()
            if root not in target.parents:  # defence in depth
                shutil.rmtree(root, ignore_errors=True)
                raise ZipValidationError(f"Unsafe extraction target: {rel!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
            report.indexed_files.append(rel)
            report.total_bytes += info.file_size

    if not report.indexed_files:
        shutil.rmtree(root, ignore_errors=True)
        raise ZipValidationError("No supported source files found in the archive.")
    report.indexed_files.sort()
    return report


def read_text(path: Path) -> str:
    """Read a file as UTF-8 text, replacing undecodable bytes."""
    return path.read_bytes().decode("utf-8", errors="replace")


def iter_repo_files(root: Path) -> list[str]:
    """List supported, non-ignored text files under ``root`` as POSIX relative paths."""
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        if is_ignored_path(rel) or language_for(rel) is None:
            continue
        try:
            with open(p, "rb") as fh:
                if looks_binary(fh.read(4096)):
                    continue
        except OSError:
            continue
        out.append(rel)
    return out
