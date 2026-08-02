"""Test vòng lặp generation phía Kaggle — chạy không cần GPU (dry-run / stub query)."""

from __future__ import annotations

import json

import pytest

from r2ai.generation.run_generation import DEFAULT_STUB_QUERY, existing_ids, materialize_csvs, run
from r2ai.schemas import RetrievalCandidate, RetrievalResult

CSV_TEXT = "Chỉ tiêu,Số cuối năm\nTiền,1.234\nĐầu tư,2.000\nTổng,3.234\n"


@pytest.fixture()
def retrieval_file(tmp_path):
    path = tmp_path / "retrieval_results.jsonl"
    lines = []
    for qid in (1, 2):
        result = RetrievalResult(
            id=qid,
            question=f"Câu hỏi {qid}?",
            tickers=["NKG"],
            years=[2022],
            scope="consolidated",
            candidates=[
                RetrievalCandidate(
                    table_ref=f"NKG_financial_statements_2022_consolidated|{qid}",
                    doc_name="NKG_financial_statements_2022_consolidated",
                    ticker="NKG",
                    year=2022,
                    scope="consolidated",
                    page=8,
                    score=10.0,
                    rank=0,
                    csv_text=CSV_TEXT,
                )
            ],
        )
        lines.append(result.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dry_run_actually_executes_sandbox(retrieval_file, tmp_path):
    """Bug 5: trước đây dry-run để query rỗng nên sandbox bị short-circuit, không test được gì."""
    out = tmp_path / "predictions.jsonl"
    stats = run(retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "exec", dry_run=True)
    assert stats["attempted"] == 2
    assert stats["exec_ok"] == 2, stats
    rows = _rows(out)
    assert all(row["pandas_query"] == DEFAULT_STUB_QUERY for row in rows)
    assert all(row["exec_error"] is None for row in rows)
    # Stub đếm số ô số ở cột cuối -> 3 dòng dữ liệu đều là số.
    assert {row["answer"] for row in rows} == {3.0}
    assert all(row["evidence"][0]["csv_path"].startswith("data/") for row in rows)


def test_dry_run_reports_sandbox_error_instead_of_empty_query(retrieval_file, tmp_path):
    out = tmp_path / "predictions.jsonl"
    run(
        retrieval_path=retrieval_file,
        out_path=out,
        work_dir=tmp_path / "exec",
        dry_run=True,
        stub_query="import os\nresult = 1",
    )
    rows = _rows(out)
    assert all(not row["exec_ok"] for row in rows)
    assert all("import" in row["exec_error"] for row in rows)


def test_stub_query_without_dry_run_skips_model(retrieval_file, tmp_path):
    """`--stub-query` cũng phải chạy được mà không nạp LLM (nếu nạp sẽ lỗi vì không có transformers)."""
    out = tmp_path / "predictions.jsonl"
    stats = run(
        retrieval_path=retrieval_file,
        out_path=out,
        work_dir=tmp_path / "exec",
        stub_query="result = float(len(df1))",
    )
    assert stats["exec_ok"] == 2
    assert {row["answer"] for row in _rows(out)} == {3.0}


def test_resume_skips_finished_ids(retrieval_file, tmp_path):
    out = tmp_path / "predictions.jsonl"
    run(retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "exec", dry_run=True, limit=1)
    assert existing_ids(out) == {1}
    stats = run(retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "exec", dry_run=True)
    assert stats["skipped_done"] == 1
    assert stats["attempted"] == 1
    assert {row["id"] for row in _rows(out)} == {1, 2}


def test_materialize_csvs_writes_expected_filenames(tmp_path):
    result = RetrievalResult(
        id=1,
        question="q",
        candidates=[
            RetrievalCandidate(
                table_ref="DOC|7", doc_name="DOC", ticker="X", csv_text=CSV_TEXT
            )
        ],
    )
    paths = materialize_csvs(result, {"df1": "DOC|7"}, tmp_path)
    assert paths["df1"].name == "DOC_table_7.csv"
    assert paths["df1"].read_text(encoding="utf-8") == CSV_TEXT
