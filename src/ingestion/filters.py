"""File filtering rules: what is indexed and what is ignored."""
from __future__ import annotations

from pathlib import PurePosixPath

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".idea", ".vscode", "node_modules", "dist", "build",
    "target", "out", ".next", ".nuxt", ".cache", ".tox", "site-packages", "coverage",
    ".gradle", "bin", "obj", ".eggs",
}

LANGUAGE_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++", ".h": "C++", ".c": "C",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML", ".md": "Markdown", ".toml": "TOML",
    ".ini": "Config", ".cfg": "Config", ".sh": "Shell", ".sql": "SQL", ".txt": "Text",
}

SPECIAL_FILENAMES = {
    "Dockerfile": "Docker", "docker-compose.yml": "YAML", "docker-compose.yaml": "YAML",
    "Makefile": "Make", ".gitignore": "Config", ".env.example": "Config",
}

# Never indexed even if the extension matches (may contain real secrets).
BLOCKED_FILENAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_dsa", "credentials"}
BLOCKED_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".min.js", ".lock")

IMPORTANT_CONFIG_FILES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "package.json",
    "tsconfig.json", "pom.xml", "build.gradle", "Dockerfile", "docker-compose.yml",
    "docker-compose.yaml", "Makefile", ".env.example", "README.md", "CMakeLists.txt",
}


def is_ignored_path(rel_path: str) -> bool:
    """True if any directory component of ``rel_path`` is in :data:`IGNORED_DIRS`."""
    parts = PurePosixPath(rel_path.replace("\\", "/")).parts
    return any(p in IGNORED_DIRS for p in parts[:-1])


def language_for(rel_path: str) -> str | None:
    """Return the language label for a path, or ``None`` if unsupported/blocked."""
    p = PurePosixPath(rel_path.replace("\\", "/"))
    if p.name in BLOCKED_FILENAMES or p.name.endswith(BLOCKED_SUFFIXES):
        return None
    if p.name in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[p.name]
    return LANGUAGE_BY_EXT.get(p.suffix.lower())


def looks_binary(data: bytes) -> bool:
    """Heuristic: NUL bytes or undecodable UTF-8 in the first 4 KB means binary."""
    head = data[:4096]
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Tolerate a multibyte character cut at the 4 KB boundary only.
        if exc.start < len(head) - 4:
            return True
    return False


def should_index(rel_path: str, size_bytes: int, max_file_kb: int, head: bytes | None = None) -> tuple[bool, str]:
    """Decide whether to index a file. Returns ``(ok, reason_if_skipped)``."""
    if is_ignored_path(rel_path):
        return False, "ignored directory"
    if language_for(rel_path) is None:
        return False, "unsupported or blocked file type"
    if size_bytes > max_file_kb * 1024:
        return False, f"larger than {max_file_kb} KB"
    if size_bytes == 0:
        return False, "empty file"
    if head is not None and looks_binary(head):
        return False, "binary content"
    return True, ""
