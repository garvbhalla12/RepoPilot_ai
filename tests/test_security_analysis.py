import logging

import pytest

from src.analysis.dashboard import run_analysis
from src.analysis.repo_summary import analyze_repository, parse_dependencies
from src.analysis.static_tools import tool_available
from src.parsing.python_ast import parse_python
from src.security.secrets import REDACTED, RedactingFilter, redact_text, scan_text

from tests.conftest import SAMPLE_REPO


def test_secret_detection_masks_values():
    fake = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    text = f'token = "{fake}"\npassword = "hunter2hunter2"\n'
    findings = scan_text("cfg.py", text)
    assert {f.rule_id for f in findings} >= {"SEC-GITHUB-TOKEN", "SEC-HARDCODED-CRED"}
    for f in findings:
        assert fake not in f.preview and "hunter2hunter2" not in f.preview
        assert REDACTED in f.preview


def test_placeholders_and_env_lookups_not_flagged():
    text = 'password = "your_password_here"\napi_key = os.getenv("API_KEY")\nsecret = "<change-me>"\n'
    assert [f for f in scan_text("a.py", text) if f.category == "secret"] == []


def test_risky_patterns_detected_and_comments_ignored():
    text = "eval(user)\nsubprocess.run(x, shell=True)\n# eval(ignored)\nhashlib.md5(b'')\n"
    ids = {f.rule_id for f in scan_text("a.py", text)}
    assert {"RISK-EVAL", "RISK-SHELL-TRUE", "RISK-WEAK-HASH"} <= ids
    assert sum(f.rule_id == "RISK-EVAL" for f in scan_text("a.py", text)) == 1


def test_log_filter_redacts(caplog):
    logger = logging.getLogger("redact-test")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger="redact-test"):
        logger.info('loaded api_key = "supersecretvalue99"')
    assert "supersecretvalue99" not in caplog.text


def test_redact_text_leaves_normal_code_untouched():
    code = "def add(a, b):\n    return a + b"
    assert redact_text(code) == code


def test_parse_python_handles_malformed_source():
    info = parse_python("def broken(:\n")
    assert info.syntax_error and not info.functions
    ok = parse_python('@app.route("/x", methods=["POST"])\nasync def h():\n    pass\n')
    assert ok.routes[0].method == "POST" and ok.routes[0].path == "/x" and ok.functions[0].is_async


def test_dependency_parsing_tolerates_bad_manifests():
    assert parse_dependencies("requirements.txt", "flask==3.0\n# c\nrequests>=2 ; python_version>'3'\n") == ["flask", "requests"]
    assert parse_dependencies("package.json", "{not json") == []
    assert "express" in parse_dependencies("package.json", '{"dependencies": {"express": "^4"}}')
    assert parse_dependencies("pyproject.toml", "[[[") == []


def test_repository_overview_on_sample():
    s = analyze_repository(SAMPLE_REPO)
    assert s.languages["Python"]["files"] >= 6 and "JavaScript" in s.languages
    assert "flask" in s.dependencies["requirements.txt"] and "Flask" in s.frameworks
    assert any(r["path"] == "/api/login" for r in s.routes)
    assert "app/" in s.tree_text.replace("└── ", "").replace("├── ", "")
    assert any(f["path"] == "tests/" and "test" in f["purpose"].lower() for f in s.folders)


def test_overview_reports_syntax_errors(tmp_path):
    (tmp_path / "bad.py").write_text("def x(:\n", encoding="utf-8")
    s = analyze_repository(tmp_path)
    assert "bad.py" in s.syntax_errors and s.python_stats["syntax_errors"] == 1


def test_heuristic_findings_in_sample_and_scanners_separate():
    rep = run_analysis(SAMPLE_REPO, use_ruff=False, use_bandit=False)
    rules = {f["rule"] for f in rep.heuristic_findings}
    assert {"RISK-EVAL", "RISK-SHELL-TRUE", "SEC-HARDCODED-CRED"} <= rules
    assert rep.scanner_findings == [] and all(f["source"] == "RepoPilot heuristic" for f in rep.heuristic_findings)


@pytest.mark.skipif(not tool_available("bandit"), reason="bandit not installed")
def test_bandit_integration():
    rep = run_analysis(SAMPLE_REPO, use_ruff=False, use_bandit=True)
    assert rep.tools[0].ran
    assert any(f["source"].startswith("Bandit") for f in rep.scanner_findings)


@pytest.mark.skipif(not tool_available("ruff"), reason="ruff not installed")
def test_ruff_integration(tmp_path):
    (tmp_path / "x.py").write_text("import os\n", encoding="utf-8")
    rep = run_analysis(tmp_path, use_ruff=True, use_bandit=False)
    assert rep.tools[0].ran and any(f["rule"] == "F401" for f in rep.scanner_findings)


def test_missing_tools_degrade_gracefully(monkeypatch, tmp_path):
    import src.analysis.static_tools as st
    monkeypatch.setattr(st, "tool_available", lambda m: False)
    r = st.run_ruff(tmp_path)
    assert not r.available and not r.ran and "not installed" in r.error
