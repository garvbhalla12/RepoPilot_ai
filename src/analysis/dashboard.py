"""Combine scanner and heuristic findings into one report (kept clearly labelled by source)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.analysis.static_tools import ToolResult, run_bandit, run_ruff
from src.ingestion.zip_loader import iter_repo_files, read_text
from src.security.secrets import scan_text

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class AnalysisReport:
    scanner_findings: list[dict] = field(default_factory=list)   # Ruff/Bandit output
    heuristic_findings: list[dict] = field(default_factory=list)  # RepoPilot rules (secrets/patterns)
    tools: list[ToolResult] = field(default_factory=list)

    def severity_counts(self) -> dict[str, int]:
        c = Counter(f["severity"] for f in self.scanner_findings + self.heuristic_findings)
        return {k: c.get(k, 0) for k in ("high", "medium", "low")}


def run_analysis(root: Path, use_ruff: bool = True, use_bandit: bool = True) -> AnalysisReport:
    """Run enabled scanners plus the built-in heuristics on ``root``."""
    report = AnalysisReport()
    if use_ruff:
        r = run_ruff(root)
        report.tools.append(r)
        report.scanner_findings.extend(r.findings)
    if use_bandit:
        b = run_bandit(root)
        report.tools.append(b)
        report.scanner_findings.extend(b.findings)
    for rel in iter_repo_files(root):
        for f in scan_text(rel, read_text(root / rel)):
            report.heuristic_findings.append({
                "source": "RepoPilot heuristic", "rule": f.rule_id, "category": f.category,
                "severity": f.severity, "message": f.message, "file": f.file_path, "line": f.line,
                "preview": f.preview, "recommendation": f.recommendation,
            })
    key = lambda f: (SEVERITY_ORDER.get(f["severity"], 3), f["file"], f["line"])  # noqa: E731
    report.scanner_findings.sort(key=key)
    report.heuristic_findings.sort(key=key)
    return report
