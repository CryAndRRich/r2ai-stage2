"""Test join retrieval + predictions -> submission.json (kèm re-execute ở local)."""

from __future__ import annotations

import json

import pytest

from r2ai.packaging.assemble_submission import assemble, relevant_tables_for
from r2ai.packaging.zip_submission import validate_build
from r2ai.schemas import EvidenceItem, Prediction, RetrievalCandidate, RetrievalResult

CSV_TEXT = "Chỉ tiêu,Số cuối năm\nTiền,1.234\nĐầu tư,2.000\n"
TABLE_REF = "NKG_financial_statements_2022_consolidated|12"


@pytest.fixture()
def inputs(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"id": 1, "question": "Tiền của NKG cuối năm 2022 là bao nhiêu đồng?"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": 2, "question": "Câu hỏi không có prediction"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result = RetrievalResult(
        id=1,
        question="Tiền của NKG cuối năm 2022 là bao nhiêu đồng?",
        tickers=["NKG"],
        years=[2022],
        scope="consolidated",
        candidates=[
            RetrievalCandidate(
                table_ref=TABLE_REF,
                doc_name="NKG_financial_statements_2022_consolidated",
                ticker="NKG",
                year=2022,
                scope="consolidated",
                page=8,
                score=12.5,
                rank=0,
                csv_text=CSV_TEXT,
            )
        ],
    )
    retrieval = tmp_path / "retrieval_results.jsonl"
    retrieval.write_text(result.model_dump_json() + "\n", encoding="utf-8")

    prediction = Prediction(
        id=1,
        pandas_query=(
            "row = df1[df1['Chỉ tiêu'] == 'Tiền']\n"
            "result = float(row['Số cuối năm'].iloc[0].replace('.', ''))\n"
        ),
        answer=1234.0,
        evidence=[EvidenceItem(variable="df1", csv_path="data/NKG_financial_statements_2022_consolidated_table_12.csv")],
        used_table_refs=[TABLE_REF],
        exec_ok=True,
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(prediction.model_dump_json() + "\n", encoding="utf-8")
    return questions, retrieval, predictions


def test_assemble_produces_valid_submission(inputs, tmp_path):
    questions, retrieval, predictions = inputs
    build_dir = tmp_path / "build"
    stats = assemble(
        questions_path=questions,
        retrieval_path=retrieval,
        predictions_path=predictions,
        build_dir=build_dir,
    )
    assert stats["items"] == 2  # câu thiếu prediction vẫn phải có item
    assert stats["with_prediction"] == 1
    assert stats["reexecuted_ok"] == 1
    assert stats["fallback_answer_used"] == 1  # id=2

    items = json.loads((build_dir / "submission.json").read_text(encoding="utf-8"))
    first = next(i for i in items if i["id"] == 1)
    assert first["answer"] == pytest.approx(1234.0)
    assert first["relevant_tables"] == [TABLE_REF]
    assert first["relevant_docs"] == ["NKG_financial_statements_2022_consolidated"]
    assert first["evidence"][0]["csv_path"].startswith("data/")
    assert (build_dir / first["evidence"][0]["csv_path"]).exists()

    second = next(i for i in items if i["id"] == 2)
    assert second["answer"] == 0.0
    assert second["pandas_query"] == ""

    assert validate_build(build_dir, questions_path=questions) == []


def test_assemble_clears_stale_csv_files(inputs, tmp_path):
    questions, retrieval, predictions = inputs
    build_dir = tmp_path / "build"
    (build_dir / "data").mkdir(parents=True)
    (build_dir / "data" / "stale_table_1.csv").write_text("x\n", encoding="utf-8")
    assemble(
        questions_path=questions,
        retrieval_path=retrieval,
        predictions_path=predictions,
        build_dir=build_dir,
    )
    assert not (build_dir / "data" / "stale_table_1.csv").exists()


def test_assemble_skips_reexecute_when_disabled(inputs, tmp_path):
    questions, retrieval, predictions = inputs
    stats = assemble(
        questions_path=questions,
        retrieval_path=retrieval,
        predictions_path=predictions,
        build_dir=tmp_path / "build",
        reexecute=False,
    )
    assert stats["reexecuted_ok"] == 0


def test_failed_local_reexecute_is_flagged_not_silent(inputs, tmp_path):
    """Bug 7.2: re-execute local hỏng thì phải ghi rõ item nào chưa được double-check."""
    questions, retrieval, predictions = inputs
    broken = Prediction(
        id=1,
        pandas_query="result = df1['Không tồn tại'].iloc[0]",
        answer=999.0,
        evidence=[
            EvidenceItem(variable="df1", csv_path="data/NKG_financial_statements_2022_consolidated_table_12.csv")
        ],
        used_table_refs=[TABLE_REF],
        exec_ok=True,
    )
    predictions.write_text(broken.model_dump_json() + "\n", encoding="utf-8")

    build_dir = tmp_path / "build"
    stats = assemble(
        questions_path=questions, retrieval_path=retrieval, predictions_path=predictions, build_dir=build_dir
    )
    assert stats["reexecuted_failed"] == 1
    assert stats["kaggle_answer_not_verified"] >= 1

    report = json.loads((build_dir / "reexec_report.json").read_text(encoding="utf-8"))
    failed = [row for row in report if row["id"] == 1 and row["status"] == "failed"]
    assert failed and failed[0]["kaggle_answer"] == 999.0

    # Answer từ Kaggle vẫn được giữ (không có gì tốt hơn) nhưng đã bị đánh dấu nghi vấn.
    items = json.loads((build_dir / "submission.json").read_text(encoding="utf-8"))
    assert next(i for i in items if i["id"] == 1)["answer"] == 999.0


def test_duplicate_evidence_variable_is_dropped(inputs, tmp_path):
    questions, retrieval, predictions = inputs
    csv_path = "data/NKG_financial_statements_2022_consolidated_table_12.csv"
    duplicated = Prediction(
        id=1,
        pandas_query="result = 1.0",
        answer=1.0,
        evidence=[
            EvidenceItem(variable="df1", csv_path=csv_path),
            EvidenceItem(variable="df1", csv_path=csv_path),
        ],
        used_table_refs=[TABLE_REF],
        exec_ok=True,
    )
    predictions.write_text(duplicated.model_dump_json() + "\n", encoding="utf-8")

    build_dir = tmp_path / "build"
    stats = assemble(
        questions_path=questions, retrieval_path=retrieval, predictions_path=predictions, build_dir=build_dir
    )
    assert stats["duplicate_evidence_variables"] == 1
    assert stats["validation_errors"] == 0
    items = json.loads((build_dir / "submission.json").read_text(encoding="utf-8"))
    variables = [e["variable"] for e in next(i for i in items if i["id"] == 1)["evidence"]]
    assert variables == ["df1"]


def test_duplicate_variable_does_not_leave_orphan_csv(inputs, tmp_path):
    """Bug 10: entry evidence bị loại vì trùng tên biến không được để lại CSV mồ côi trong data/."""
    questions, retrieval, predictions = inputs
    result = RetrievalResult.model_validate_json(retrieval.read_text(encoding="utf-8").strip())
    other = result.candidates[0].model_copy(
        update={"table_ref": "NKG_financial_statements_2022_consolidated|99", "line": 99, "rank": 1}
    )
    retrieval.write_text(
        result.model_copy(update={"candidates": [*result.candidates, other]}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    duplicated = Prediction(
        id=1,
        pandas_query="result = 1.0",
        answer=1.0,
        evidence=[
            EvidenceItem(variable="df1", csv_path="data/NKG_financial_statements_2022_consolidated_table_12.csv"),
            EvidenceItem(variable="df1", csv_path="data/NKG_financial_statements_2022_consolidated_table_99.csv"),
        ],
        used_table_refs=[TABLE_REF, "NKG_financial_statements_2022_consolidated|99"],
        exec_ok=True,
    )
    predictions.write_text(duplicated.model_dump_json() + "\n", encoding="utf-8")

    build_dir = tmp_path / "build"
    stats = assemble(
        questions_path=questions, retrieval_path=retrieval, predictions_path=predictions, build_dir=build_dir
    )
    assert stats["duplicate_evidence_variables"] == 1
    written = sorted(p.name for p in (build_dir / "data").glob("*.csv"))
    assert written == ["NKG_financial_statements_2022_consolidated_table_12.csv"]

    items = json.loads((build_dir / "submission.json").read_text(encoding="utf-8"))
    referenced = {e["csv_path"].split("/")[-1] for item in items for e in item["evidence"]}
    assert referenced == set(written)  # không file nào mồ côi


def test_assemble_self_validation_reports_zero_errors(inputs, tmp_path):
    questions, retrieval, predictions = inputs
    stats = assemble(
        questions_path=questions,
        retrieval_path=retrieval,
        predictions_path=predictions,
        build_dir=tmp_path / "build",
    )
    assert stats["validation_errors"] == 0


def test_only_evidence_tables_get_csv_files(inputs, tmp_path):
    """relevant_tables chỉ cần id; chỉ bảng nào evidence dùng mới được ghi CSV vào zip."""
    questions, retrieval, predictions = inputs
    # Thêm 2 candidate nữa -> relevant_tables có 3 id nhưng evidence chỉ dùng 1.
    result = RetrievalResult.model_validate_json(retrieval.read_text(encoding="utf-8").strip())
    extra = [
        result.candidates[0].model_copy(
            update={
                "table_ref": f"NKG_financial_statements_2022_consolidated|{n}",
                "rank": i + 1,
                "line": n,
            }
        )
        for i, n in enumerate((40, 77))
    ]
    retrieval.write_text(
        result.model_copy(update={"candidates": [*result.candidates, *extra]}).model_dump_json() + "\n",
        encoding="utf-8",
    )

    build_dir = tmp_path / "build"
    assemble(
        questions_path=questions, retrieval_path=retrieval, predictions_path=predictions, build_dir=build_dir
    )
    items = json.loads((build_dir / "submission.json").read_text(encoding="utf-8"))
    first = next(i for i in items if i["id"] == 1)
    assert len(first["relevant_tables"]) == 3  # vẫn nộp đủ 3 id cho retrieval F2
    assert len(first["evidence"]) == 1
    csv_files = sorted(p.name for p in (build_dir / "data").glob("*.csv"))
    assert csv_files == ["NKG_financial_statements_2022_consolidated_table_12.csv"]


def test_relevant_docs_cover_all_relevant_tables(inputs, tmp_path):
    questions, retrieval, predictions = inputs
    build_dir = tmp_path / "build"
    assemble(
        questions_path=questions, retrieval_path=retrieval, predictions_path=predictions, build_dir=build_dir
    )
    items = json.loads((build_dir / "submission.json").read_text(encoding="utf-8"))
    for item in items:
        docs = {ref.rsplit("|", 1)[0] for ref in item["relevant_tables"]}
        assert docs <= set(item["relevant_docs"])


def test_relevant_tables_puts_used_tables_first():
    result = RetrievalResult(
        id=1,
        question="q",
        candidates=[
            RetrievalCandidate(table_ref="d|1", doc_name="d", ticker="X"),
            RetrievalCandidate(table_ref="d|2", doc_name="d", ticker="X"),
        ],
    )
    prediction = Prediction(id=1, used_table_refs=["d|2"])
    assert relevant_tables_for(result, prediction, n_tables=2) == ["d|2", "d|1"]
    assert relevant_tables_for(result, None, n_tables=1) == ["d|1"]
