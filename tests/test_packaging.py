"""Test validate cấu trúc submission theo COMPETITION.md mục 3 + zip đúng layout."""

from __future__ import annotations

import json
import zipfile

import pytest

from r2ai.packaging.zip_submission import SubmissionInvalidError, validate_build, zip_submission


def _item(qid: int = 1, **overrides) -> dict:
    item = {
        "id": qid,
        "question": "Doanh thu thuần của VNM năm 2023 là bao nhiêu?",
        "answer": 63075000000.0,
        "relevant_docs": ["VNM_financial_statements_2023_consolidated"],
        "relevant_tables": ["VNM_financial_statements_2023_consolidated|350"],
        "evidence": [
            {"variable": "df1", "csv_path": "data/VNM_financial_statements_2023_consolidated_table_350.csv"}
        ],
        "pandas_query": "result = 1.0",
    }
    item.update(overrides)
    return item


@pytest.fixture()
def build_dir(tmp_path):
    root = tmp_path / "submission"
    (root / "data").mkdir(parents=True)
    (root / "data" / "VNM_financial_statements_2023_consolidated_table_350.csv").write_text(
        "a,b\n1,2\n", encoding="utf-8"
    )
    (root / "submission.json").write_text(json.dumps([_item()]), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "x"}) + "\n", encoding="utf-8")
    return root, questions


def test_valid_build_has_no_errors(build_dir):
    root, questions = build_dir
    assert validate_build(root, questions_path=questions) == []


def test_missing_question_id_is_error(build_dir):
    root, questions = build_dir
    questions.write_text(
        json.dumps({"id": 1, "question": "x"}) + "\n" + json.dumps({"id": 2, "question": "y"}) + "\n",
        encoding="utf-8",
    )
    errors = validate_build(root, questions_path=questions)
    assert any("thiếu 1 câu hỏi" in e for e in errors)


def test_absolute_csv_path_is_error(build_dir):
    root, questions = build_dir
    item = _item(evidence=[{"variable": "df1", "csv_path": "/tmp/x.csv"}])
    (root / "submission.json").write_text(json.dumps([item]), encoding="utf-8")
    assert any("data/" in e for e in validate_build(root, questions_path=questions))


def test_duplicate_variable_in_same_question_is_error(build_dir):
    root, questions = build_dir
    csv_path = "data/VNM_financial_statements_2023_consolidated_table_350.csv"
    item = _item(
        evidence=[{"variable": "df1", "csv_path": csv_path}, {"variable": "df1", "csv_path": csv_path}]
    )
    (root / "submission.json").write_text(json.dumps([item]), encoding="utf-8")
    assert any("trùng" in e for e in validate_build(root, questions_path=questions))


def test_invalid_variable_name_is_error(build_dir):
    root, questions = build_dir
    item = _item(
        evidence=[{"variable": "1df", "csv_path": "data/VNM_financial_statements_2023_consolidated_table_350.csv"}]
    )
    (root / "submission.json").write_text(json.dumps([item]), encoding="utf-8")
    assert any("identifier" in e for e in validate_build(root, questions_path=questions))


def test_missing_referenced_csv_is_error(build_dir):
    root, questions = build_dir
    (root / "data" / "VNM_financial_statements_2023_consolidated_table_350.csv").unlink()
    assert any("thiếu file CSV" in e for e in validate_build(root, questions_path=questions))


def test_answer_must_be_number(build_dir):
    root, questions = build_dir
    (root / "submission.json").write_text(json.dumps([_item(answer="63 tỷ")]), encoding="utf-8")
    assert any("`answer`" in e for e in validate_build(root, questions_path=questions))


def test_bad_table_ref_format_is_error(build_dir):
    root, questions = build_dir
    item = _item(relevant_tables=["VNM_financial_statements_2023_consolidated_table_350"])
    (root / "submission.json").write_text(json.dumps([item]), encoding="utf-8")
    assert any("table_ref" in e for e in validate_build(root, questions_path=questions))


def test_relevant_docs_must_cover_relevant_tables(build_dir):
    root, questions = build_dir
    (root / "submission.json").write_text(json.dumps([_item(relevant_docs=[])]), encoding="utf-8")
    assert any("relevant_docs" in e for e in validate_build(root, questions_path=questions))


def test_duplicate_ids_is_error(build_dir):
    root, questions = build_dir
    (root / "submission.json").write_text(json.dumps([_item(1), _item(1)]), encoding="utf-8")
    assert any("trùng id" in e for e in validate_build(root, questions_path=questions))


def test_extra_json_file_is_error(build_dir):
    root, questions = build_dir
    (root / "extra.json").write_text("{}", encoding="utf-8")
    assert any("1 file .json" in e for e in validate_build(root, questions_path=questions))


def test_internal_report_files_do_not_count_as_extra_json(build_dir):
    root, questions = build_dir
    (root / "assemble_stats.json").write_text("{}", encoding="utf-8")
    (root / "reexec_report.json").write_text("[]", encoding="utf-8")
    assert validate_build(root, questions_path=questions) == []


def test_validate_without_questions_path_skips_id_coverage(build_dir):
    root, _ = build_dir
    assert validate_build(root) == []  # không truyền questions -> không kiểm tra đủ id


def test_zip_layout_has_no_parent_folder(build_dir, tmp_path):
    root, _ = build_dir
    (root / "assemble_stats.json").write_text("{}", encoding="utf-8")
    zip_path = zip_submission(root, tmp_path / "submission.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    assert names == [
        "data/VNM_financial_statements_2023_consolidated_table_350.csv",
        "submission.json",
    ]


def test_zip_submission_self_validates_when_called_directly(build_dir, tmp_path):
    """Bug 7.1: gọi thẳng `zip_submission()` (bỏ qua main) trước đây không validate gì cả."""
    root, _ = build_dir
    (root / "data" / "VNM_financial_statements_2023_consolidated_table_350.csv").unlink()
    zip_path = tmp_path / "submission.zip"
    with pytest.raises(SubmissionInvalidError):
        zip_submission(root, zip_path)
    assert not zip_path.exists()


def test_zip_submission_checks_id_coverage_when_questions_given(build_dir, tmp_path):
    root, questions = build_dir
    questions.write_text(
        json.dumps({"id": 1, "question": "x"}) + "\n" + json.dumps({"id": 2, "question": "y"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SubmissionInvalidError):
        zip_submission(root, tmp_path / "submission.zip", questions_path=questions)
