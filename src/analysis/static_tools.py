"""Optional Ruff and Bandit integration (run as subprocesses; code is never executed)."""
from __future__ import annotations

import importlib.util
import json
import subprocess  # nosec B404 - fixed argv only, no shell
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolResult:
    tool: str
    available: bool
    ran: bool = False
    findings: list[dict] = field(default_factory=list)
    error: str = ""


def tool_available(module: str) -> bool:
    """True if the Python module for a tool (``ruff``/``bandit``) is importable."""
    return importlib.util.find_spec(module) is not None


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603  # nosec B603 - fixed argv list, no shell, no user code run
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False,
    )


def _ruff_severity(code: str) -> str:
    if code.startswith("E9") or code in {"F821", "F822", "F823"}:
        return "high"
    return "medium" if code.startswith("F") else "low"


def run_ruff(root: Path, timeout: int = 90) -> ToolResult:
    """Run ``ruff check`` in isolated mode (ignores repo config) and normalise findings."""
    res = ToolResult("ruff", tool_available("ruff"))
    if not res.available:
        res.error = "Ruff is not installed (python -m pip install ruff)."
        return res
    try:
        proc = _run([sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache",
                     "--output-format", "json", "."], root, timeout)
        items = json.loads(proc.stdout or "[]")
    except subprocess.TimeoutExpired:
        res.error = f"Ruff timed out after {timeout}s."
        return res
    except (json.JSONDecodeError, OSError) as exc:
        res.error = f"Ruff failed: {exc}"
        return res
    res.ran = True
    for it in items:
        code = it.get("code") or "ruff"
        res.findings.append({
            "source": "Ruff (scanner)", "rule": code, "severity": _ruff_severity(code),
            "message": it.get("message", ""),
            "file": Path(it.get("filename", "")).as_posix().replace(root.as_posix() + "/", ""),
            "line": (it.get("location") or {}).get("row", 0),
            "recommendation": "See https://docs.astral.sh/ruff/rules/ for the rule description.",
        })
    return res


def run_bandit(root: Path, timeout: int = 120) -> ToolResult:
    """Run Bandit recursively and normalise findings. Bandit exits non-zero when it finds issues."""
    res = ToolResult("bandit", tool_available("bandit"))
    if not res.available:
        res.error = "Bandit is not installed (python -m pip install bandit)."
        return res
    try:
        proc = _run([sys.executable, "-m", "bandit", "-r", ".", "-f", "json", "-q"], root, timeout)
        data = json.loads(proc.stdout or "{}")
    except subprocess.TimeoutExpired:
        res.error = f"Bandit timed out after {timeout}s."
        return res
    except (json.JSONDecodeError, OSError) as exc:
        res.error = f"Bandit failed or produced no JSON output: {exc}"
        return res
    res.ran = True
    for it in data.get("results", []):
        res.findings.append({
            "source": "Bandit (scanner)", "rule": it.get("test_id", "bandit"),
            "severity": str(it.get("issue_severity", "LOW")).lower(),
            "message": it.get("issue_text", ""),
            "file": Path(it.get("filename", "")).as_posix().lstrip("./"),
            "line": it.get("line_number", 0),
            "recommendation": it.get("more_info", "") or "See Bandit documentation.",
        })
    return res
