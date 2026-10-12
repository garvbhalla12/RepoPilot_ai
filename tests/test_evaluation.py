import csv
import json

from src.evaluation.runner import export_report, load_dataset, run_evaluation

from tests.conftest import ROOT, SAMPLE_REPO

DATASET = ROOT / "data" / "evaluation" / "questions.json"


def test_dataset_is_valid_and_references_real_files():
    qs = load_dataset(DATASET)
    assert len(qs) >= 10
    for q in qs:
        for f in q["relevant_files"]:
            assert (SAMPLE_REPO / f).exists(), f"{q['id']} references missing file {f}"
        text = "".join((SAMPLE_REPO / f).read_text(encoding="utf-8") for f in q["relevant_files"]).lower()
        for kw in q["expected_keywords"]:  # labels must be satisfiable from the labelled files
            assert kw.lower() in text, f"{q['id']}: keyword {kw!r} not in labelled files"


def test_evaluation_runs_without_api_key_and_exports(settings, store, embedder, tmp_path):
    rep = run_evaluation(settings, embedder, store, SAMPLE_REPO, DATASET, k=5)
    assert rep.meta["llm_used"] is False
    assert rep.summary["failed_queries"] == 0
    for key in ("precision@5", "recall@5", "mrr", "citation_file_correctness", "keyword_coverage"):
        assert 0.0 <= rep.summary[key] <= 1.0
    assert rep.summary["latency_ms_mean"] > 0
    jp, cp = export_report(rep, tmp_path / "out")
    assert json.loads(jp.read_text())["summary"]["questions_total"] == len(rep.rows)
    with cp.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(rep.rows) and "human_rating" in rows[0]


def test_evaluation_is_reproducible(settings, store, embedder):
    a = run_evaluation(settings, embedder, store, SAMPLE_REPO, DATASET, k=5).summary
    b = run_evaluation(settings, embedder, store, SAMPLE_REPO, DATASET, k=5).summary
    for key in ("precision@5", "recall@5", "mrr", "keyword_coverage"):
        assert a[key] == b[key]
