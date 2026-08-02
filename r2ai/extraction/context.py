"""Ngữ cảnh quanh mỗi bảng: số trang gần nhất + vài dòng text ngay trước bảng.

Vài dòng trước bảng thường chứa tiêu đề bảng ("BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG") và dòng đơn vị
tính ("Đơn vị: VND") — tín hiệu quan trọng cho cả BM25 lẫn prompt sinh pandas.
"""

from __future__ import annotations

import bisect
import re

from r2ai.constants import PAGE_MARKER_PATTERN

_PAGE_RE = re.compile(PAGE_MARKER_PATTERN)
_MARKER_LINE_RE = re.compile(rf"^\s*{PAGE_MARKER_PATTERN}\s*$")


class DocumentContext:
    """Tra cứu page/context theo offset ký tự trong text nguồn (O(log n) mỗi truy vấn)."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._offsets: list[int] = []
        self._pages: list[int] = []
        for match in _PAGE_RE.finditer(text):
            self._offsets.append(match.start())
            self._pages.append(int(match.group(1)))

    def page_at(self, offset: int) -> int | None:
        """Số trang của marker `===== PAGE n =====` gần nhất phía trước `offset`."""
        pos = bisect.bisect_right(self._offsets, offset) - 1
        return self._pages[pos] if pos >= 0 else None

    def text_before(self, offset: int, *, max_lines: int = 3, max_chars: int = 400) -> str:
        """Lấy tối đa `max_lines` dòng text không rỗng ngay trước `offset`.

        Bỏ qua dòng marker trang và dòng chỉ chứa đuôi của một bảng khác (`</table>`).
        """
        window = self._text[max(0, offset - 4000) : offset]
        lines: list[str] = []
        for raw in reversed(window.splitlines()):
            line = raw.strip()
            if not line or _MARKER_LINE_RE.match(line):
                continue
            if line.endswith("</table>") or line.startswith("<table"):
                break
            lines.append(line)
            if len(lines) >= max_lines:
                break
        return " ".join(reversed(lines))[-max_chars:]
