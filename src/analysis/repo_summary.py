"""Repository overview: tree, languages, dependencies, folders, Python structure.

Folder purposes and framework guesses are *heuristic estimates* based on names and
imports; the UI labels them as such.
"""
from __future__ import annotations

import json
import re
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.filters import IMPORTANT_CONFIG_FILES, language_for
from src.ingestion.zip_loader import iter_repo_files, read_text
from src.parsing.python_ast import PyModuleInfo, parse_python

FOLDER_HINTS = {
    "src": "Main source code", "app": "Application code", "lib": "Library/shared code",
    "tests": "Automated tests", "test": "Automated tests", "__tests__": "Automated tests",
    "docs": "Documentation", "doc": "Documentation", "scripts": "Helper scripts",
    "config": "Configuration", "configs": "Configuration", "models": "Data models / ORM",
    "routes": "HTTP route handlers", "api": "API layer", "controllers": "Request controllers",
    "services": "Business logic", "utils": "Utility helpers", "helpers": "Utility helpers",
    "static": "Static assets", "templates": "HTML templates", "public": "Public web assets",
    "components": "UI components", "pages": "UI pages/views", "views": "Views",
    "migrations": "Database migrations", "db": "Database access", "database": "Database access",
    "middleware": "Request middleware", "auth": "Authentication", "data": "Data / fixtures",
    "assets": "Static assets", "styles": "Stylesheets", "hooks": "Hooks / lifecycle code",
    "core": "Core logic", "backend": "Backend service", "frontend": "Frontend client",
}

FRAMEWORK_IMPORTS = {
    "flask": "Flask", "fastapi": "FastAPI", "django": "Django", "streamlit": "Streamlit",
    "sqlalchemy": "SQLAlchemy", "pydantic": "Pydantic", "pytest": "pytest", "requests": "requests",
    "numpy": "NumPy", "pandas": "pandas", "torch": "PyTorch", "tensorflow": "TensorFlow",
}


@dataclass
class RepoSummary:
    root: str
    total_files: int = 0
    total_lines: int = 0
    languages: dict[str, dict[str, int]] = field(default_factory=dict)  # lang -> {files, lines}
    extensions: dict[str, int] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)  # manifest -> names
    config_files: list[str] = field(default_factory=list)
    folders: list[dict] = field(default_factory=list)  # {path, files, purpose}
    frameworks: list[str] = field(default_factory=list)
    routes: list[dict] = field(default_factory=list)
    python_stats: dict[str, int] = field(default_factory=dict)
    syntax_errors: dict[str, str] = field(default_factory=dict)
    tree_text: str = ""


def parse_dependencies(name: str, text: str) -> list[str]:
    """Extract dependency names from a manifest. Returns ``[]`` on malformed input."""
    deps: list[str] = []
    try:
        if name == "requirements.txt":
            for line in text.splitlines():
                line = line.split("#")[0].strip()
                if line and not line.startswith(("-", "git+", "http")):
                    deps.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
        elif name == "package.json":
            data = json.loads(text)
            for key in ("dependencies", "devDependencies"):
                deps.extend((data.get(key) or {}).keys())
        elif name == "pyproject.toml":
            data = tomllib.loads(text)
            for d in (data.get("project", {}).get("dependencies") or []):
                deps.append(re.split(r"[<>=!~\[; ]", d, maxsplit=1)[0])
            poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            deps.extend(k for k in poetry if k.lower() != "python")
        elif name == "pom.xml":
            deps.extend(re.findall(r"<artifactId>([^<]+)</artifactId>", text))
    except (ValueError, TypeError, AttributeError):
        return []
    return sorted({d for d in deps if d})


def build_tree_text(files: list[str], max_lines: int = 400) -> str:
    """Render a sorted ASCII tree from relative file paths."""
    tree: dict = {}
    for f in files:
        node = tree
        for part in f.split("/"):
            node = node.setdefault(part, {})
    lines: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        items = sorted(node.items(), key=lambda kv: (not kv[1], kv[0].lower()))
        for i, (name, child) in enumerate(items):
            if len(lines) >= max_lines:
                return
            last = i == len(items) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}{'/' if child else ''}")
            if child:
                walk(child, prefix + ("    " if last else "│   "))

    walk(tree, "")
    if len(lines) >= max_lines:
        lines.append("... (truncated)")
    return "\n".join(lines)


def analyze_repository(root: Path) -> RepoSummary:
    """Analyse a repository directory. Python files are parsed with ``ast`` only."""
    files = iter_repo_files(root)
    s = RepoSummary(root=str(root), total_files=len(files))
    lang_files: Counter[str] = Counter()
    lang_lines: Counter[str] = Counter()
    ext: Counter[str] = Counter()
    folder_files: dict[str, int] = defaultdict(int)
    imports: Counter[str] = Counter()
    py_funcs = py_classes = py_files = 0

    for rel in files:
        p = Path(rel)
        lang = language_for(rel) or "Other"
        text = read_text(root / rel)
        lines = len(text.splitlines())
        lang_files[lang] += 1
        lang_lines[lang] += lines
        s.total_lines += lines
        ext[p.suffix.lower() or p.name] += 1
        if len(p.parts) > 1:
            folder_files[p.parts[0]] += 1
        if p.name in IMPORTANT_CONFIG_FILES:
            s.config_files.append(rel)
        if p.name in {"requirements.txt", "package.json", "pyproject.toml", "pom.xml"}:
            deps = parse_dependencies(p.name, text)
            if deps:
                s.dependencies[rel] = deps
        if lang == "Python":
            py_files += 1
            info: PyModuleInfo = parse_python(text)
            if info.syntax_error:
                s.syntax_errors[rel] = info.syntax_error
                continue
            py_funcs += len(info.functions)
            py_classes += len(info.classes)
            imports.update(info.imports)
            for r in info.routes:
                s.routes.append({"method": r.method, "path": r.path, "handler": r.handler, "file": rel, "line": r.line})

    s.languages = {lang: {"files": lang_files[lang], "lines": lang_lines[lang]} for lang, _ in lang_files.most_common()}
    s.extensions = dict(ext.most_common())
    s.frameworks = sorted({FRAMEWORK_IMPORTS[i] for i in imports if i in FRAMEWORK_IMPORTS})
    s.folders = [
        {"path": f + "/", "files": n, "purpose": FOLDER_HINTS.get(f.lower(), "Not recognised (no estimate)")}
        for f, n in sorted(folder_files.items(), key=lambda kv: -kv[1])
    ]
    s.python_stats = {"python_files": py_files, "functions": py_funcs, "classes": py_classes, "syntax_errors": len(s.syntax_errors)}
    s.tree_text = build_tree_text(files)
    return s
