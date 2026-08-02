"""Test parse HTML bảng: colspan/rowspan, dòng lệch cột, HTML lỗi do OCR."""

from __future__ import annotations

from r2ai.extraction.html_tables import (
    check_tag_balance,
    clean_cell,
    grid_to_csv,
    locate_tables,
    parse_table,
    split_header,
)


def test_locate_tables_counts_and_indexes_1_based():
    text = "abc <table><tr><td>1</td></tr></table> def <table><tr><td>2</td></tr></table>"
    tables = locate_tables(text)
    assert [t.index for t in tables] == [1, 2]
    assert [t.line for t in tables] == [1, 1]  # cùng dòng vì text 1 dòng
    assert tables[0].html.endswith("</table>")
    assert text[tables[1].start : tables[1].end] == tables[1].html


def test_locate_tables_unbalanced_does_not_crash(caplog):
    text = "<table><tr><td>a</td></tr></table><table><tr><td>b</td></tr>"
    tables = locate_tables(text, source="broken.txt")
    assert check_tag_balance(text) == (2, 1)
    # Bảng cuối thiếu </table> -> giữ chỗ (placeholder rỗng), không bị bỏ khỏi dãy số thứ tự.
    assert [t.index for t in tables] == [1, 2]
    assert tables[0].closed and tables[1].closed is False
    assert tables[1].html == ""
    assert parse_table(tables[1].html) == []


def test_unclosed_table_does_not_swallow_the_next_one():
    """Bug 3: regex non-greedy cũ gộp bảng lỗi với bảng kế tiếp -> mọi bảng sau lệch chỉ số 1."""
    text = (
        "intro\n"
        "<table><tr><td>A1</td></tr>\n"          # thiếu </table>
        "text ở giữa\n"
        "<table><tr><td>B1</td></tr></table>\n"
        "<table><tr><td>C1</td></tr></table>\n"
    )
    tables = locate_tables(text, source="broken.txt")
    assert [t.index for t in tables] == [1, 2, 3]
    assert [t.closed for t in tables] == [False, True, True]
    grids = [parse_table(t.html) for t in tables]
    assert grids == [[], [["B1"]], [["C1"]]]  # B và C giữ đúng vị trí 2 và 3


def test_line_numbers_are_1_indexed():
    """`<vị trí>` của table_ref = số dòng bắt đầu bảng (BTC xác nhận), đếm từ 1."""
    text = "dòng 1\ndòng 2\n<table><tr><td>A</td></tr></table>\ndòng 4\n<table><tr><td>B</td></tr></table>\n"
    assert [t.line for t in locate_tables(text)] == [3, 5]


def test_line_numbers_survive_multiline_tables():
    text = "\n".join(["header", "<table>", "<tr><td>A</td></tr>", "</table>", "giữa", "<table><tr><td>B</td></tr></table>"])
    assert [t.line for t in locate_tables(text)] == [2, 6]


def test_unclosed_table_in_the_middle_keeps_following_indexes():
    text = (
        "<table><tr><td>A1</td></tr></table>"
        "<table><tr><td>B1</td></tr>"           # thiếu </table>
        "<table><tr><td>C1</td></tr></table>"
    )
    tables = locate_tables(text)
    assert [(t.index, t.closed) for t in tables] == [(1, True), (2, False), (3, True)]
    assert parse_table(tables[2].html) == [["C1"]]


def test_parse_table_simple_grid():
    html = "<table><tr><td>Mã số</td><td>TÀI SẢN</td></tr><tr><td>100</td><td>1.000</td></tr></table>"
    grid = parse_table(html)
    assert grid == [["Mã số", "TÀI SẢN"], ["100", "1.000"]]
    header, rows = split_header(grid)
    assert header == ["Mã số", "TÀI SẢN"]
    assert rows == [["100", "1.000"]]


def test_parse_table_colspan_repeats_value():
    html = "<table><tr><td colspan='3'>Năm 2020</td></tr><tr><td>a</td><td>b</td><td>c</td></tr></table>"
    assert parse_table(html) == [["Năm 2020", "Năm 2020", "Năm 2020"], ["a", "b", "c"]]


def test_parse_table_rowspan_carries_down_correct_column():
    # Case thật trong corpus: cột "Thuyết minh" dùng rowspan=3 (AAA 2020 separate).
    html = (
        "<table>"
        "<tr><td>110</td><td>Tiền</td><td rowspan='3'>4</td><td>816</td></tr>"
        "<tr><td>111</td><td>Tiền mặt</td><td>179</td></tr>"
        "<tr><td>112</td><td>Tương đương tiền</td><td>438</td></tr>"
        "</table>"
    )
    grid = parse_table(html)
    assert grid == [
        ["110", "Tiền", "4", "816"],
        ["111", "Tiền mặt", "4", "179"],
        ["112", "Tương đương tiền", "4", "438"],
    ]
    # Giá trị số luôn nằm ở cột cuối -> colspan/rowspan không làm lệch cột.
    assert [row[-1] for row in grid] == ["816", "179", "438"]


def test_parse_table_pads_short_rows():
    html = "<table><tr><td>a</td><td>b</td><td>c</td></tr><tr><td>x</td></tr></table>"
    assert parse_table(html) == [["a", "b", "c"], ["x", "", ""]]


def test_parse_table_clamps_absurd_span():
    html = "<table><tr><td colspan='99999'>x</td></tr></table>"
    grid = parse_table(html)
    assert 1 <= len(grid[0]) <= 40


def test_parse_table_tolerates_broken_ocr_markup():
    html = "<table><tr><td>a<td>b</tr><tr><td>c</td><td>d</table>"
    grid = parse_table(html)
    assert grid and grid[0][0] == "a"


def test_parse_table_ignores_empty_rows():
    html = "<table><tr><td></td><td></td></tr><tr><td>a</td><td>1</td></tr></table>"
    assert parse_table(html) == [["a", "1"]]


def test_parse_table_returns_empty_for_no_rows():
    assert parse_table("<table></table>") == []


def test_clean_cell_normalizes_whitespace_and_nbsp():
    assert clean_cell("  Tiền và  tương\nđương ") == "Tiền và tương đương"


def test_grid_to_csv_quotes_commas():
    csv_text = grid_to_csv([["a,b", "c"], ["1", "2"]])
    assert csv_text == '"a,b",c\n1,2\n'
