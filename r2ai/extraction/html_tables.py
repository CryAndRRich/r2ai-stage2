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


HEADER_JOIN = " | "  # phân cách giữa các tầng header khi gộp -> "Số cuối năm | Giá trị"
MAX_HEADER_ROWS = 3  # tầng 1 (colspan) + tầng 2 (nhãn con) + tầng 3 (dòng đơn vị) là mức sâu nhất gặp thật
_MAX_HEADER_CELL_CHARS = 60  # ô header là nhãn ngắn; ô dài gần như luôn là câu văn của dòng dữ liệu
_PERIOD_TOKEN_RE = re.compile(
    r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}"  # 31/12/2024
    r"|\d{1,2}\s*[/\-.]\s*(?:19|20)\d{2}"  # 12/2024
    r"|[QT]\s*\d"  # Q1, T3
    r"|(?:19|20)\d{2}"  # 2024
)
_UNIT_ONLY_RE = re.compile(
    r"^(?:đơn\s*vị\s*(?:tính)?\s*[:\-]?\s*)?"
    r"(?:(?:nghìn\s+tỷ|nghìn|triệu|tỷ)\s*)?"
    r"(?:vnd|đồng|vn\s*đ|usd|eur|jpy|%)$",
    re.IGNORECASE,
)


def _duplicate_positions(header: list[str]) -> list[int]:
    """Vị trí các cột có tên (không rỗng) trùng với ≥1 cột khác — dấu hiệu colspan đã bị làm phẳng."""
    counts: dict[str, int] = {}
    for cell in header:
        if cell:
            counts[cell] = counts.get(cell, 0) + 1
    return [i for i, cell in enumerate(header) if cell and counts[cell] > 1]


def _looks_numeric_cell(cell: str) -> bool:
    """Ô còn chữ số sau khi bỏ các token kỳ (năm/ngày/quý) -> gần như chắc chắn là ô dữ liệu.

    Không dùng "có chữ số" trần: nhãn header hợp lệ rất hay chứa năm hoặc ngày
    ("31/12/2024", "Khấu hao TSCĐ Năm 2024", "Q1/2024"). Ngược lại một ô số liệu thật
    ("39.562.950.995", "2,92%") luôn còn lại chữ số sau khi bỏ token kỳ.
    """
    return bool(re.search(r"\d", _PERIOD_TOKEN_RE.sub(" ", cell)))


def _is_unit_row(row: list[str]) -> bool:
    """Dòng chỉ gồm token đơn vị ("Triệu VND", "%", "Đơn vị tính: VND") -> tầng header đơn vị."""
    cells = [c.strip() for c in row if c.strip()]
    return len(cells) >= 2 and all(_UNIT_ONLY_RE.match(c) for c in cells)


def _is_header_continuation(header: list[str], row: list[str]) -> bool:
    """`row` có phải tầng header tiếp theo của `header` (chứ không phải dòng dữ liệu đầu tiên)?

    Điều kiện (cố ý chặt — nhận nhầm 1 dòng dữ liệu thành header là mất luôn 1 dòng số liệu):
    1. `header` đang có tên cột trùng lặp (hệ quả của colspan bị làm phẳng) — trừ khi `row` là
       dòng đơn vị thuần, trường hợp đó vẫn nên gộp để đơn vị nằm trong tên cột.
    2. `row` không chứa ô số liệu nào (nhãn kỳ dạng năm/ngày không tính là số).
    3. `row` phủ đủ các cột đang trùng tên và ô ở đó không rỗng — tức nó thật sự mang thông tin
       phân biệt cho đúng những cột đã mất thông tin.
    4. Mọi ô của `row` đều ngắn (nhãn header), không phải câu văn dài của dòng dữ liệu.
    """
    if not row or not any(cell.strip() for cell in row):
        return False
    if any(_looks_numeric_cell(cell) for cell in row):
        return False
    if any(len(cell) > _MAX_HEADER_CELL_CHARS for cell in row):
        return False
    if _is_unit_row(row):
        return True
    duplicates = _duplicate_positions(header)
    if not duplicates:
        return False
    if any(i >= len(row) or not row[i].strip() for i in duplicates):
        return False
    # Phải phân biệt được: nếu mọi cột trùng tên vẫn nhận cùng một nhãn con thì gộp cũng vô ích
    # (và rủi ro ăn nhầm dòng dữ liệu toàn text giống nhau do rowspan carry).
    return len({row[i] for i in duplicates}) >= 2


def _merge_header_row(header: list[str], row: list[str]) -> list[str]:
    """Gộp 1 tầng header vào tên cột hiện có; bỏ phần trùng lặp để không ra "Tên | Tên"."""
    width = max(len(header), len(row))
    merged: list[str] = []
    for i in range(width):
        top = header[i].strip() if i < len(header) else ""
        sub = row[i].strip() if i < len(row) else ""
        if not sub or sub == top or sub in top.split(HEADER_JOIN):
            merged.append(top)
        elif not top:
            merged.append(sub)
        else:
            merged.append(f"{top}{HEADER_JOIN}{sub}")
    return merged


def split_header(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Tách header khỏi dữ liệu, **gộp header nhiều tầng** thành một dòng tên cột duy nhất.

    Bug 14: bản v1 luôn lấy đúng `grid[0]`. Với bảng header 2 tầng (tầng 1 "Số cuối năm"/"Số đầu năm"
    dùng colspan, tầng 2 "Giá trị"/"Dự phòng"), colspan expansion làm tầng 1 lặp giá trị nên header
    CSV có **tên cột trùng nhau y hệt** (đo thật: 870/4.770 bảng = 18,2% bài nộp V1), còn tầng 2 —
    nơi chứa thông tin phân biệt — rơi xuống thành dòng dữ liệu đầu tiên. Model không thể chọn đúng
    cột niên độ/kỳ từ header như vậy, nên đây là điều kiện tiên quyết để sửa Bug 15.

    Gộp tối đa `MAX_HEADER_ROWS` tầng và chỉ khi dòng sau thật sự **trông như header** — xem
    `_is_header_continuation`. Grid trong cache không đổi (hàm này chạy lúc đọc), nên không cần
    extract lại corpus; chỉ cần build lại `tables_index.jsonl` + retrieval.
    """
    if not grid:
        return [], []
    header = list(grid[0])
    consumed = 1
    while consumed < len(grid) - 1 and consumed < MAX_HEADER_ROWS:
        if not _is_header_continuation(header, grid[consumed]):
            break
        header = _merge_header_row(header, grid[consumed])
        consumed += 1
    return header, [list(row) for row in grid[consumed:]]
