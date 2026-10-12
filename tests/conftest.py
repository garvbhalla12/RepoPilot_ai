"""Shared fixtures. Tests use the offline hashing embedder so no model download is needed."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Settings  # noqa: E402
from src.embeddings.embedder import HashingEmbedder  # noqa: E402
from src.retrieval.indexer import index_repository  # noqa: E402
from src.retrieval.store import VectorStore  # noqa: E402

SAMPLE_REPO = ROOT / "data" / "sample_repo"


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return Settings(data_dir=tmp_path / "data", llm_enabled=False)


@pytest.fixture()
def embedder():
    return HashingEmbedder()


@pytest.fixture()
def store(tmp_path):
    return VectorStore(tmp_path / "chroma")


@pytest.fixture()
def indexed(store, embedder):
    index_repository(SAMPLE_REPO, "sample", store, embedder)
    return store
