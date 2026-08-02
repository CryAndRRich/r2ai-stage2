"""Định vị & parse bảng HTML nhúng trong file OCR `.txt`.

Thiết kế theo ARCHITECTURE.md mục 3:
- Định vị bằng regex `<table>...</table>` (DOTALL), log cảnh báo khi tag lệch thay vì crash.
- Parse bằng `lxml.html` (khoan dung với tag lỗi OCR) và mở rộng colspan/rowspan thành lưới
  hình chữ nhật, sau đó pad các dòng thiếu cột.
"""

from __future__ import annotations

import bisect
import csv
import io
import logging
import re
import unicodedata
from dataclasses import dataclass

import lxml.html

from r2ai.constants import MAX_SPAN, MAX_TABLE_CELLS

logger = logging.getLogger(__name__)

_OPEN_TAG_RE = re.compile(r"<table\b", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class RawTable:
    """Vị trí thô của một bảng trong text nguồn."""

    index: int  # 1-indexed theo thứ tự xuất hiện trong file
    start: int
    end: int
    html: str
    closed: bool = True  # False = thiếu `</table>` -> placeholder giữ chỗ, html rỗng
    line: int = 0  # số dòng 1-indexed nơi tag `<table` bắt đầu -> dùng làm `<vị trí>` trong table_ref


def check_tag_balance(text: str) -> tuple[int, int]:
    """Trả về (số tag mở, số tag đóng) — dùng để phát hiện file OCR lệch tag."""
    return len(_OPEN_TAG_RE.findall(text)), len(_CLOSE_TAG_RE.findall(text))


def line_starts(text: str) -> list[int]:
    """Offset ký tự bắt đầu của mỗi dòng — để tra số dòng bằng bisect thay vì đếm `\\n` mỗi lần."""
    offsets = [0]
    start = text.find("\n")
    while start != -1:
        offsets.append(start + 1)
        start = text.find("\n", start + 1)
    return offsets


def line_of(offsets: list[int], position: int) -> int:
    """Số dòng **1-indexed** chứa `position`."""
    return bisect.bisect_right(offsets, position)


def locate_tables(text: str, *, source: str = "<memory>") -> list[RawTable]:
    """Quét bảng theo **từng tag `<table` mở**, không dùng regex `<table>...</table>` một phát.

    Mỗi bảng có 2 số: `index` (thứ tự xuất hiện, 1-indexed) và `line` (số dòng bắt đầu, 1-indexed).
    **`line` mới là `<vị trí>` dùng trong `table_ref`** theo xác nhận của BTC ở mục discussion:
    "vị trí bảng ở đây là số line bắt đầu bảng trong file ocr báo cáo tương ứng".

    Vì sao không dùng regex non-greedy: nếu một bảng bị OCR làm mất `</table>`, regex
    `<table\\b[^>]*>.*?</table>` sẽ **nuốt luôn bảng kế tiếp** thành 1 match, làm mọi bảng phía
    sau lệch chỉ số 1 đơn vị (và grid gộp trông vẫn "hợp lý" nên lỗi đi qua im lặng). Ở đây mỗi
    tag mở luôn chiếm đúng 1 vị trí trong dãy số thứ tự: bảng thiếu tag đóng trở thành
    placeholder (`closed=False`, `html=""`) thay vì nuốt bảng sau.
    """
    opens = [m.start() for m in _OPEN_TAG_RE.finditer(text)]
    closes = [m.end() for m in _CLOSE_TAG_RE.finditer(text)]
    if len(opens) != len(closes):
        logger.warning(
            "tag <table> lệch trong %s: open=%d close=%d (bảng thiếu tag đóng vẫn giữ chỗ)",
            source,
            len(opens),
            len(closes),
        )

    offsets = line_starts(text)
    tables: list[RawTable] = []
    next_close = 0
    for i, start in enumerate(opens):
        index = i + 1
        line = line_of(offsets, start)
        boundary = opens[i + 1] if i + 1 < len(opens) else len(text)
        while next_close < len(closes) and closes[next_close] <= start:
            next_close += 1
        end = closes[next_close] if next_close < len(closes) else None
        if end is None or end > boundary:
            # Không có `</table>` trước tag mở kế tiếp -> bảng lỗi, giữ chỗ bằng placeholder rỗng.
            logger.warning(
                "bảng thứ %d (dòng %d) trong %s thiếu tag đóng </table> — giữ chỗ bằng bảng rỗng",
                index,
                line,
                source,
            )
            tables.append(RawTable(index=index, start=start, end=boundary, html="", closed=False, line=line))
            continue
        tables.append(RawTable(index=index, start=start, end=end, html=text[start:end], line=line))
        next_close += 1
    return tables


def clean_cell(text: str) -> str:
    """Chuẩn hoá text trong 1 ô: NFC, bỏ nbsp, gộp whitespace."""
    normalized = unicodedata.normalize("NFC", text).replace(" ", " ")
    return _WS_RE.sub(" ", normalized).strip()


def _span(value: str | None) -> int:
    """Đọc colspan/rowspan; OCR có thể sinh giá trị rác nên clamp về [1, MAX_SPAN]."""
    if not value:
        return 1
    match = re.search(r"\d+", value)
    if not match:
        return 1
    return max(1, min(int(match.group(0)), MAX_SPAN))


def parse_table(html: str, *, source: str = "<memory>") -> list[list[str]]:
    """Parse 1 bảng HTML -> lưới hình chữ nhật (list[list[str]]).

    Ô gộp (colspan/rowspan) được lặp lại giá trị sang các cột/dòng bị span. Dòng thiếu cột
    được pad bằng chuỗi rỗng cho bằng số cột lớn nhất. Trả về `[]` nếu không parse được
    (kể cả placeholder rỗng của bảng thiếu tag đóng — xem `locate_tables`).
    """
    if not html.strip():
        return []
    try:
        root = lxml.html.fromstring(html)
    except Exception as exc:  # lxml có thể ném ParserError/XMLSyntaxError trên rác OCR
        logger.warning("không parse được bảng trong %s: %s", source, exc)
        return []

    tr_nodes = root.xpath(".//tr")
    if not tr_nodes:
        return []

    grid: list[list[str]] = []
    # pending[col] = (value, số dòng còn phải lặp lại) — carry của rowspan.
    pending: dict[int, tuple[str, int]] = {}
    total_cells = 0

    for tr in tr_nodes:
        cells = tr.xpath("./td|./th")
        row: list[str] = []
        col = 0
        next_cell = 0
        while True:
            carry = pending.get(col)
            if carry is not None and carry[1] > 0:
                # Cột này đang bị rowspan của dòng trên chiếm -> lặp lại giá trị.
                row.append(carry[0])
                pending[col] = (carry[0], carry[1] - 1)
                col += 1
                continue
            if next_cell >= len(cells):
                break
            cell = cells[next_cell]
            next_cell += 1
            value = clean_cell(cell.text_content() or "")
            colspan = _span(cell.get("colspan"))
            rowspan = _span(cell.get("rowspan"))
            for _ in range(colspan):
                row.append(value)
                if rowspan > 1:
                    pending[col] = (value, rowspan - 1)
                col += 1

        # Carry còn lại ở các cột nằm sau ô cuối cùng của dòng này.
        for carry_col in sorted(c for c, state in pending.items() if c >= col and state[1] > 0):
            value, remaining = pending[carry_col]
            while len(row) < carry_col:
                row.append("")
            row.append(value)
            pending[carry_col] = (value, remaining - 1)

        pending = {c: state for c, state in pending.items() if state[1] > 0}
        grid.append(row)
        total_cells += len(row)
        if total_cells > MAX_TABLE_CELLS:
            logger.warning(
                "bảng trong %s vượt %d ô — cắt bớt (nghi lỗi span/OCR)", source, MAX_TABLE_CELLS
            )
            break

    grid = [row for row in grid if any(cell for cell in row)]
    if not grid:
        return []
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


def grid_to_csv(grid: list[list[str]]) -> str:
    """Serialize lưới thành CSV (LF, quote khi cần) — đúng dạng nộp ở `evidence[].csv_path`."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(grid)
    return buffer.getvalue()


def split_header(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Coi dòng 0 là header (không heuristic phức tạp ở v1)."""
    if not grid:
        return [], []
    return grid[0], grid[1:]
