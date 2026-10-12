"""Run the retrieval evaluation from the command line (no API key required).

Examples (PowerShell):
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --backend hashing --k 5
    python scripts/run_evaluation.py --with-llm      # only if LLM_ENABLED=true and a key is set
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from src.embeddings.embedder import EmbeddingUnavailableError, get_embedder  # noqa: E402
from src.evaluation.runner import export_report, run_evaluation  # noqa: E402
from src.retrieval.store import VectorStore  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="RepoPilot AI retrieval evaluation")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--backend", choices=["sentence-transformers", "hashing"], default=None,
                   help="Embedding backend (default: EMBEDDING_BACKEND from .env)")
    p.add_argument("--with-llm", action="store_true", help="Also generate answers (needs an enabled provider)")
    p.add_argument("--out", type=Path, default=None, help="Report directory (default: <data_dir>/reports)")
    args = p.parse_args()

    s = get_settings()
    s.ensure_dirs()
    backend = args.backend or s.embedding_backend
    try:
        embedder = get_embedder(backend, s.embedding_model)
    except EmbeddingUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="repopilot_eval_") as tmp:  # isolated store; never touches user indexes
        store = VectorStore(Path(tmp) / "chroma")
        report = run_evaluation(s, embedder, store, ROOT / "data" / "sample_repo",
                                ROOT / "data" / "evaluation" / "questions.json", k=args.k, with_llm=args.with_llm)
        del store
    jp, cp = export_report(report, args.out or s.reports_dir)

    print(f"Embedder: {report.meta['embedder']}   k={args.k}   LLM used: {report.meta['llm_used']}")
    for key, val in report.summary.items():
        print(f"  {key:34s} {val}")
    print(f"\n{report.meta['disclaimer']}\nWrote:\n  {jp}\n  {cp}")
    return 0 if report.summary["failed_queries"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
