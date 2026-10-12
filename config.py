"""Central configuration for RepoPilot AI.

All settings come from environment variables (optionally loaded from a local
``.env`` file) with safe defaults. No secret is ever hardcoded here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings. Instantiate via :func:`get_settings`."""

    # Storage
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("REPOPILOT_DATA_DIR", ROOT_DIR / ".repopilot_data")))

    # Upload limits (ZIP validation)
    max_zip_mb: int = field(default_factory=lambda: _env_int("MAX_ZIP_MB", 50))
    max_files: int = field(default_factory=lambda: _env_int("MAX_FILES", 5000))
    max_extracted_mb: int = field(default_factory=lambda: _env_int("MAX_EXTRACTED_MB", 200))
    max_file_kb: int = field(default_factory=lambda: _env_int("MAX_FILE_KB", 512))

    # Embeddings: "sentence-transformers" (default) or "hashing" (offline, no model download)
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "sentence-transformers"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    # Chunking / retrieval
    chunk_max_lines: int = field(default_factory=lambda: _env_int("CHUNK_MAX_LINES", 60))
    chunk_overlap_lines: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_LINES", 10))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    min_score: float = field(default_factory=lambda: float(os.getenv("MIN_SCORE", "0.05")))

    # Optional LLM provider. Disabled unless LLM_ENABLED=true AND a key is set.
    llm_enabled: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", False))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))  # openai | anthropic
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 60))
    max_question_chars: int = field(default_factory=lambda: _env_int("MAX_QUESTION_CHARS", 2000))

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def llm_api_key(self) -> str:
        """Read the key lazily from the environment; never stored on the object."""
        name = "ANTHROPIC_API_KEY" if self.llm_provider == "anthropic" else "OPENAI_API_KEY"
        return os.getenv(name, "")

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.chroma_dir, self.repos_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a fresh Settings object (reads the environment each call)."""
    return Settings()
