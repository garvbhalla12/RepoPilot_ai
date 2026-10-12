"""Reproducible evaluation pipeline over the bundled sample repository."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from config import Settings
from src.embeddings.embedder import Embedder
from src.evaluation import metrics as M
from src.rag import llm
from src.rag.pipeline import RAGPipeline
from src.retrieval.indexer import index_repository
from src.retrieval.search import semantic_search
from src.retrieval.store import VectorStore

EVAL_REPO_ID = "eval-sample"
HUMAN_NOTE = (
    "Precision@K, Recall@K, MRR, citation-file correctness and keyword coverage are AUTOMATED PROXY metrics "
    "computed against labels written by the project author. They are not human-verified answer quality. "
    "Fill the 'human_rating' column of the CSV to record manual review."
)


@dataclass
class EvalReport:
    meta: dict
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def load_dataset(path: Path) -> list[dict]:
    """Load and validate the question set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data["questions"]
    for q in qs:
        if not {"id", "question", "relevant_files", "expected_keywords", "answerable"} <= q.keys():
            raise ValueError(f"Malformed evaluation question: {q.get('id')}")
    return qs


def run_evaluation(
    settings: Settings,
    embedder: Embedder,
    store: VectorStore,
    repo_dir: Path,
    dataset_path: Path,
    k: int = 5,
    with_llm: bool = False,
    reindex: bool = True,
) -> EvalReport:
    """Index the sample repo, run each question, and aggregate metrics. Works without any API key."""
    questions = load_dataset(dataset_path)
    if reindex or not store.exists(EVAL_REPO_ID):
        index_repository(repo_dir, EVAL_REPO_ID, store, embedder, settings.chunk_max_lines, settings.chunk_overlap_lines)
    ready, reason = llm.llm_status(settings)
    use_llm = with_llm and ready
    pipeline = RAGPipeline(settings, store, embedder)

    rows: list[dict] = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
               "relevant_files": ";".join(q["relevant_files"]), "error": "", "human_rating": ""}
        t0 = time.perf_counter()
        try:
            hits = semantic_search(store, embedder, EVAL_REPO_ID, q["question"], k=k)
            row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            files = [h.file_path for h in hits]
            row["retrieved_files"] = ";".join(dict.fromkeys(files))
            row["top1_file"] = files[0] if files else ""
            row["max_score"] = hits[0].score if hits else 0.0
            if q["answerable"]:
                row["precision_at_k"] = round(M.precision_at_k(files, q["relevant_files"], k), 4)
                row["recall_at_k"] = round(M.recall_at_k(files, q["relevant_files"], k), 4)
                row["reciprocal_rank"] = round(M.reciprocal_rank(files, q["relevant_files"]), 4)
                row["citation_file_correct"] = round(M.citation_file_correctness(files, q["relevant_files"], k), 4)
                row["keyword_coverage"] = round(M.keyword_coverage([h.text for h in hits], q["expected_keywords"]), 4)
            else:
                row["abstained"] = bool(not hits or hits[0].score < settings.min_score)
            if use_llm:
                res = pipeline.ask(q["question"], EVAL_REPO_ID, k=k, use_llm=True)
                row["mode"] = res.mode
                row["grounding_passes"] = bool(res.grounding and res.grounding["passes"])
                row["total_latency_ms"] = res.total_ms
                row["error"] = res.error or ""
                row["answer"] = res.answer or ""
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        rows.append(row)

    ans = [r for r in rows if r["answerable"] and not r["error"]]
    unans = [r for r in rows if not r["answerable"] and not r["error"]]
    lat = [r["latency_ms"] for r in rows if "latency_ms" in r]
    summary = {
        f"precision@{k}": round(mean(r["precision_at_k"] for r in ans), 4) if ans else None,
        f"recall@{k}": round(mean(r["recall_at_k"] for r in ans), 4) if ans else None,
        "mrr": round(M.mean_reciprocal_rank([r["reciprocal_rank"] for r in ans]), 4) if ans else None,
        "citation_file_correctness": round(mean(r["citation_file_correct"] for r in ans), 4) if ans else None,
        "keyword_coverage": round(mean(r["keyword_coverage"] for r in ans), 4) if ans else None,
        "top1_file_accuracy": round(mean(1.0 if r["top1_file"] in r["relevant_files"].split(";") else 0.0 for r in ans), 4) if ans else None,
        "abstention_rate_on_unanswerable": round(mean(1.0 if r.get("abstained") else 0.0 for r in unans), 4) if unans else None,
        "latency_ms_mean": round(mean(lat), 2) if lat else None,
        "latency_ms_p95": round(M.percentile(lat, 95), 2) if lat else None,
        "questions_total": len(rows),
        "questions_answerable": len([r for r in rows if r["answerable"]]),
        "failed_queries": len([r for r in rows if r["error"]]),
        "llm_grounding_pass_rate": (round(mean(1.0 if r.get("grounding_passes") else 0.0 for r in rows if r["answerable"] and "grounding_passes" in r), 4)
                                    if use_llm and any("grounding_passes" in r for r in rows) else None),
    }
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k, "embedder": embedder.name, "dataset": dataset_path.name,
        "llm_used": use_llm, "llm_note": "" if use_llm else (reason if with_llm else "LLM not requested; retrieval-only evaluation."),
        "chunk_max_lines": settings.chunk_max_lines, "chunk_overlap_lines": settings.chunk_overlap_lines,
        "min_score": settings.min_score, "disclaimer": HUMAN_NOTE,
    }
    return EvalReport(meta=meta, rows=rows, summary=summary)


def export_report(report: EvalReport, out_dir: Path, stem: str = "evaluation") -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.csv`` to ``out_dir``. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, cp = out_dir / f"{stem}.json", out_dir / f"{stem}.csv"
    jp.write_text(json.dumps({"meta": report.meta, "summary": report.summary, "rows": report.rows}, indent=2), encoding="utf-8")
    cols: list[str] = []
    for r in report.rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with cp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(report.rows)
    return jp, cp
