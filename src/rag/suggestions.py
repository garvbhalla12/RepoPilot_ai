"""AI-generated remediation suggestions. Labelled separately from real scanner output."""
from __future__ import annotations

from config import Settings
from src.rag import llm

SYSTEM = (
    "You are a secure-coding assistant. You receive a list of static-analysis findings (rule id, message, file, line). "
    "No source code is provided. Give brief, practical remediation advice per rule. Do not claim the findings are "
    "confirmed vulnerabilities and do not claim a comprehensive audit."
)


def suggest_fixes(settings: Settings, findings: list[dict], limit: int = 20) -> str:
    """Ask the configured LLM for remediation advice. Sends only rule ids/messages/locations (no code, no secrets)."""
    lines = [f"- {f['rule']} ({f['severity']}): {f['message']} at {f['file']}:{f['line']}" for f in findings[:limit]]
    if not lines:
        return "No findings to explain."
    return llm.generate(settings, SYSTEM, "Findings:\n" + "\n".join(lines))
