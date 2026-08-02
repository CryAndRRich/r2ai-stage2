"""Test end-to-end bước extraction trên một corpus giả nhỏ (scan -> cache -> tables_index.jsonl)."""

from __future__ import annotations

import json

import pytest

from r2ai.constants import SCOPE_CONSOLIDATED, SCOPE_SEPARATE
from r2ai.extraction.build_table_index import build, is_eligible
from r2ai.extraction.context import DocumentContext
from r2ai.extraction.doc_scanner import doc_name_from_path, scan_documents
from r2ai.extraction.table_store import csv_filename, load_table, load_table_csv, parse_table_ref

DOC_TEXT = """===== PAGE 1 =====
Công ty Cổ phần Nhựa An Phát Xanh
Báo cáo tài chính riêng

===== PAGE 9 =====
BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG
Đơn vị: VND

<table><tr><td>Mã số</td><td>TÀI SẢN</td><td>Thuyết minh</td><td>Số cuối năm</td></tr>\
<tr><td>110</td><td>Tiền</td><td rowspan="2">4</td><td>816.523.338.816</td></tr>\
<tr><td>111</td><td>Tiền mặt</td><td>179.620.574.162</td></tr></table>

===== PAGE 10 =====
BAN KIỂM SOÁT

<table><tr><td>Bà Nguyễn Thị Giang</td><td>Trưởng ban</td></tr></table>
"""


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "financial_statements"
    for ticker, year, doc in [
        ("AAA", "2020", "AAA_financial_statements_2020_separate"),
        ("AAA", "2020", "AAA_financial_statements_2020_consolidated"),
    ]:
        doc_dir = root / ticker / year / doc
        doc_dir.mkdir(parents=True)
        (doc_dir / f"{doc}_extracted.txt").write_text(DOC_TEXT, encoding="utf-8")
    company_meta = tmp_path / "code_stock.csv"
    company_meta.write_text("Mã CK,Tên công ty\nAAA,CTCP Nhựa An Phát Xanh\n", encoding="utf-8")
    return root, company_meta


def test_doc_name_strips_extracted_suffix(tmp_path):
    path = tmp_path / "AAA_financial_statements_2020_separate_extracted.txt"
    assert doc_name_from_path(path) == "AAA_financial_statements_2020_separate"


def test_scan_documents_extracts_metadata(corpus):
    root, _ = corpus
    docs = scan_documents(root)
    assert [d.doc_name for d in docs] == [
        "AAA_financial_statements_2020_consolidated",
        "AAA_financial_statements_2020_separate",
    ]
    assert {d.scope for d in docs} == {SCOPE_CONSOLIDATED, SCOPE_SEPARATE}
    assert {d.year for d in docs} == {2020}
    assert {d.ticker for d in docs} == {"AAA"}


def test_document_context_tracks_pages_and_preceding_text():
    context = DocumentContext(DOC_TEXT)
    table_offset = DOC_TEXT.index("<table>")
    assert context.page_at(table_offset) == 9
    before = context.text_before(table_offset)
    assert "Đơn vị: VND" in before
    assert "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG" in before


def test_is_eligible_filters_tiny_and_text_only_tables():
    numeric = [["Chỉ tiêu", "Số"], ["Tiền", "1.000"]]
    text_only = [["Bà Nguyễn Thị Giang", "Trưởng ban"], ["Bà Nguyễn Thị Phương", "Thành viên"]]
    assert is_eligible(numeric, min_cells=4, require_numeric=True)
    assert not is_eligible(text_only, min_cells=4, require_numeric=True)
    assert not is_eligible([["x"]], min_cells=4, require_numeric=False)


def test_build_writes_index_and_cache(corpus, tmp_path):
    root, company_meta = corpus
    cache_dir = tmp_path / "cache"
    index_path = tmp_path / "tables_index.jsonl"
    stats = build(
        statements_dir=root,
        cache_dir=cache_dir,
        index_path=index_path,
        company_meta_path=company_meta,
    )
    assert stats["documents"] == 2
    assert stats["documents_extracted"] == 2
    assert stats["tables_located"] == 4  # 2 bảng/report
    # Bảng nhân sự (không có ô số) bị loại khỏi index, bảng cân đối kế toán thì không.
    assert stats["tables_indexed"] == 2

    # `<vị trí>` = số dòng bắt đầu bảng trong file OCR (dòng 8 của DOC_TEXT), KHÔNG phải thứ tự bảng.
    table_line = DOC_TEXT.splitlines().index("<table><tr><td>Mã số</td><td>TÀI SẢN</td><td>Thuyết minh</td><td>Số cuối năm</td></tr>"
                                             "<tr><td>110</td><td>Tiền</td><td rowspan=\"2\">4</td><td>816.523.338.816</td></tr>"
                                             "<tr><td>111</td><td>Tiền mặt</td><td>179.620.574.162</td></tr></table>") + 1
    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert {e["table_ref"] for e in entries} == {
        f"AAA_financial_statements_2020_consolidated|{table_line}",
        f"AAA_financial_statements_2020_separate|{table_line}",
    }
    entry = entries[0]
    assert entry["line"] == table_line
    assert entry["order"] == 1  # vẫn giữ thứ tự xuất hiện làm thông tin phụ
    assert entry["page"] == 9
    assert "CTCP Nhựa An Phát Xanh" in entry["index_text"]
    assert "Đơn vị: VND" in entry["index_text"]
    assert entry["n_cols"] == 4

    table = load_table("AAA_financial_statements_2020_separate", table_line, cache_dir=cache_dir)
    assert table is not None
    assert table.header == ["Mã số", "TÀI SẢN", "Thuyết minh", "Số cuối năm"]
    assert table.rows[1] == ["111", "Tiền mặt", "4", "179.620.574.162"]  # rowspan lan xuống đúng cột


def test_build_uses_cache_on_second_run(corpus, tmp_path):
    root, company_meta = corpus
    cache_dir = tmp_path / "cache"
    index_path = tmp_path / "tables_index.jsonl"
    kwargs = {"statements_dir": root, "cache_dir": cache_dir, "index_path": index_path, "company_meta_path": company_meta}
    build(**kwargs)
    second = build(**kwargs)
    assert second["documents_from_cache"] == 2
    assert second["documents_extracted"] == 0
    third = build(**kwargs, rebuild=True)
    assert third["documents_extracted"] == 2


def test_load_table_csv_and_ref_helpers(corpus, tmp_path):
    root, company_meta = corpus
    cache_dir = tmp_path / "cache"
    build(
        statements_dir=root,
        cache_dir=cache_dir,
        index_path=tmp_path / "tables_index.jsonl",
        company_meta_path=company_meta,
    )
    line = json.loads((tmp_path / "tables_index.jsonl").read_text(encoding="utf-8").splitlines()[0])["line"]
    csv_text = load_table_csv("AAA_financial_statements_2020_separate", line, cache_dir=cache_dir)
    assert csv_text.splitlines()[0] == "Mã số,TÀI SẢN,Thuyết minh,Số cuối năm"

    assert parse_table_ref("AAA_x_2020_separate|350") == ("AAA_x_2020_separate", 350)
    assert csv_filename("AAA_x_2020_separate|350") == "AAA_x_2020_separate_table_350.csv"
    with pytest.raises(ValueError):
        parse_table_ref("AAA_x_2020_separate")


def test_load_table_csv_truncates_at_line_boundary(corpus, tmp_path):
    root, company_meta = corpus
    cache_dir = tmp_path / "cache"
    build(
        statements_dir=root,
        cache_dir=cache_dir,
        index_path=tmp_path / "tables_index.jsonl",
        company_meta_path=company_meta,
    )
    line = json.loads((tmp_path / "tables_index.jsonl").read_text(encoding="utf-8").splitlines()[0])["line"]
    csv_text = load_table_csv(
        "AAA_financial_statements_2020_separate", line, cache_dir=cache_dir, max_chars=30
    )
    assert csv_text.endswith("\n") and len(csv_text) <= 31
