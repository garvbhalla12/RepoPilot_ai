"""Heuristic secret and risky-pattern detection with masking.

IMPORTANT: this is a rule-based heuristic, not a security audit. It produces false
positives and false negatives. Matched secret values are never stored, displayed or
logged; findings carry only a redacted preview.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

REDACTED = "[REDACTED]"

# (rule id, description, compiled regex, severity). Group "val" (if present) is the secret.
SECRET_RULES: list[tuple[str, str, re.Pattern[str], str]] = [
    ("SEC-AWS-KEY", "Possible AWS access key ID", re.compile(r"(?P<val>\bAKIA[0-9A-Z]{16}\b)"), "high"),
    ("SEC-PRIVATE-KEY", "Private key block", re.compile(r"(?P<val>-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----)"), "high"),
    ("SEC-GITHUB-TOKEN", "Possible GitHub token", re.compile(r"(?P<val>\bgh[pousr]_[A-Za-z0-9]{30,}\b)"), "high"),
    ("SEC-SLACK-TOKEN", "Possible Slack token", re.compile(r"(?P<val>\bxox[abprs]-[A-Za-z0-9-]{10,}\b)"), "high"),
    ("SEC-API-KEY-PREFIX", "Possible API key (sk- prefix)", re.compile(r"(?P<val>\bsk-[A-Za-z0-9_-]{20,}\b)"), "high"),
    ("SEC-JWT", "Possible JSON Web Token", re.compile(r"(?P<val>\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b)"), "medium"),
    (
        "SEC-HARDCODED-CRED", "Hardcoded credential-like assignment",
        re.compile(
            r"(?i)(?<![A-Za-z0-9])(?:password|passwd|pwd|secret|api[_-]?key|apikey|access[_-]?token|auth[_-]?token|client[_-]?secret)(?![A-Za-z0-9])"
            r"[\"']?\s*[:=]\s*[\"'](?P<val>[^\"'\s]{8,})[\"']"
        ),
        "medium",
    ),
]

_PLACEHOLDER = re.compile(
    r"(?i)(your[_-]|<.*>|\$\{|%\(|changeme|change_me|example|placeholder|xxxx|\*{4,}|dummy|sample|test[_-]?key|todo|none|null|env\b|getenv|environ)"
)

RISKY_RULES: list[tuple[str, str, re.Pattern[str], str, str]] = [
    ("RISK-EVAL", "Use of eval()", re.compile(r"\beval\s*\("), "high", "Avoid eval on dynamic input; use ast.literal_eval or explicit parsing."),
    ("RISK-EXEC", "Use of exec()", re.compile(r"\bexec\s*\("), "high", "Avoid exec; restructure to call known functions."),
    ("RISK-SHELL-TRUE", "subprocess with shell=True", re.compile(r"shell\s*=\s*True"), "high", "Pass an argument list and omit shell=True."),
    ("RISK-OS-SYSTEM", "os.system call", re.compile(r"\bos\.system\s*\("), "medium", "Use subprocess.run with an argument list."),
    ("RISK-PICKLE", "pickle deserialisation", re.compile(r"\bpickle\.loads?\s*\("), "high", "Do not unpickle untrusted data; prefer JSON."),
    ("RISK-YAML-LOAD", "yaml.load without safe loader", re.compile(r"\byaml\.load\s*\((?![^)]*SafeLoader)"), "medium", "Use yaml.safe_load."),
    ("RISK-WEAK-HASH", "MD5/SHA1 usage", re.compile(r"\bhashlib\.(md5|sha1)\s*\("), "low", "Use SHA-256+ for security purposes (and bcrypt/argon2 for passwords)."),
    ("RISK-TLS-VERIFY", "TLS verification disabled", re.compile(r"verify\s*=\s*False"), "medium", "Keep certificate verification enabled."),
    ("RISK-DEBUG", "Debug mode enabled", re.compile(r"\bdebug\s*=\s*True\b"), "low", "Disable debug mode in production."),
    ("RISK-SQL-FORMAT", "SQL built with string formatting", re.compile(r"(?i)\b(select|insert|update|delete)\b[^\n]*\b(from|into|set|where)\b[^\n]*(%s|\{[^}]*\}|\"\s*\+|'\s*\+|%\s*\()", ), "medium", "Use parameterised queries."),
    ("RISK-JS-EVAL", "eval/Function in JavaScript", re.compile(r"\bnew Function\s*\("), "medium", "Avoid dynamic code construction."),
    ("RISK-INNERHTML", "innerHTML assignment", re.compile(r"\.innerHTML\s*="), "medium", "Use textContent or sanitise HTML to prevent XSS."),
]


@dataclass(frozen=True)
class Finding:
    """One heuristic finding. ``preview`` is always redacted."""

    rule_id: str
    category: str  # "secret" | "risky-pattern"
    severity: str
    message: str
    file_path: str
    line: int
    preview: str
    recommendation: str
    source: str = "RepoPilot heuristic"


def _is_placeholder(value: str, line: str) -> bool:
    return bool(_PLACEHOLDER.search(value)) or bool(_PLACEHOLDER.search(line.split("=")[-1] if "=" in line else line))


def redact_text(text: str) -> str:
    """Redact anything in ``text`` that matches a secret rule (for UI, logs and LLM prompts)."""
    for _rid, _desc, rx, _sev in SECRET_RULES:
        def _sub(m: re.Match[str]) -> str:
            start, end = m.span("val")
            s0, _ = m.span()
            return m.group(0)[: start - s0] + REDACTED + m.group(0)[end - s0:]
        text = rx.sub(_sub, text)
    return text


def _preview(line: str) -> str:
    return redact_text(line.strip())[:160]


def scan_text(file_path: str, text: str, *, check_risky: bool = True) -> list[Finding]:
    """Scan one file's text. Never raises on odd input."""
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(line) > 2000:
            continue
        for rid, desc, rx, sev in SECRET_RULES:
            m = rx.search(line)
            if not m:
                continue
            if rid == "SEC-HARDCODED-CRED" and _is_placeholder(m.group("val"), line):
                continue
            findings.append(Finding(
                rid, "secret", sev, desc, file_path, lineno, _preview(line),
                "Remove the value from source, rotate it if real, and load it from environment variables or a secret manager.",
            ))
        if check_risky and not file_path.endswith((".md", ".txt", ".json", ".lock")):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*")):
                continue
            for rid, desc, rx, sev, rec in RISKY_RULES:
                if rx.search(line):
                    findings.append(Finding(rid, "risky-pattern", sev, desc, file_path, lineno, _preview(line), rec))
    return findings


class RedactingFilter(logging.Filter):
    """Logging filter that redacts secret-like substrings from every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_text(record.getMessage())
            record.args = ()
        except Exception:  # nosec B110 - a log filter must never raise
            pass
        return True


def install_log_redaction() -> None:
    """Attach :class:`RedactingFilter` to the root logger's handlers (idempotent)."""
    root = logging.getLogger()
    if not any(isinstance(f, RedactingFilter) for f in root.filters):
        root.addFilter(RedactingFilter())
    for h in root.handlers:
        if not any(isinstance(f, RedactingFilter) for f in h.filters):
            h.addFilter(RedactingFilter())
