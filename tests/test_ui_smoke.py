"""Headless UI smoke tests using Streamlit's AppTest (no browser). They check that every page
renders and the main workflows execute without exceptions; they do not test visual appearance."""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import ROOT  # noqa: E402

PAGES = ["🏠 Home", "📤 Upload", "🗂️ Overview", "🔎 Semantic search", "💬 Chat", "🛡️ Static analysis", "📊 Evaluation", "⚙️ Settings"]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("LLM_ENABLED", "false")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    return at


def go(at, page):
    at.radio(key="nav").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def click(at, text):
    btn = next(b for b in at.button if text in b.label)
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_every_page_renders_with_empty_state(app):
    assert not app.exception
    for p in PAGES:
        go(app, p)


def test_sample_workflow_end_to_end(app):
    click(app, "Load bundled sample")
    go(app, "📤 Upload")
    click(app, "Index")
    assert any("Indexed" in s.value for s in app.success)
    go(app, "🗂️ Overview")
    go(app, "🔎 Semantic search")
    app.text_input[0].set_value("how are passwords hashed").run()
    assert not app.exception and any("app/auth.py" in e.label for e in app.expander)
    go(app, "💬 Chat")
    assert any("No model provider enabled" in m.value for m in app.markdown)
    app.chat_input[0].set_value("How are passwords hashed?").run()
    assert not app.exception
    assert any("Retrieval only" in m.value for m in app.markdown)
    go(app, "🛡️ Static analysis")
    click(app, "Run analysis")
    go(app, "📊 Evaluation")
    click(app, "Run evaluation")
    assert any(m.label.startswith("precision@") for m in app.metric)
    go(app, "⚙️ Settings")


def test_search_without_index_shows_warning(app):
    click(app, "Load bundled sample")
    go(app, "🔎 Semantic search")
    assert any("no index yet" in w.value for w in app.warning)
