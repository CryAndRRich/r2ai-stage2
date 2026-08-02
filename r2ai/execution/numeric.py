"""Parse số kiểu Việt Nam ('.'=nghìn, ','=thập phân, '(x)'=âm, '%').

⚠️ CHỦ Ý: các hàm ở đây **không** được inject vào namespace sandbox. Lý do: BTC có thể tự
re-execute `pandas_query` trong môi trường của họ (Execution Accuracy), nơi không tồn tại helper
của mình — query phụ thuộc helper sẽ fail. Vì vậy prompt yêu cầu LLM viết code tự chứa, còn
module này chỉ dùng cho phân tích/kiểm tra nội bộ (sanity probe, so sánh đáp án, unit test).
"""

from __future__ import annotations

import math
import re
import unicodedata

from r2ai.constants import ANSWER_ABS_TOL

_NUMERIC_CHARS_RE = re.compile(r"[\d.,\-+()%\s]+")
_DIGIT_RE = re.compile(r"\d")


def parse_vn_number(value: object) -> float | None:
    """'1.234.567' -> 1234567.0 ; '12,5' -> 12.5 ; '(1.234)' -> -1234.0 ; '' -> None.

    Trả về None khi không phải số (nhãn chỉ tiêu, ô rỗng, text lẫn số như 'Ghi chú 5').
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)

    text = unicodedata.normalize("NFC", str(value)).replace(" ", " ").strip()
    if not text or not _DIGIT_RE.search(text):
        return None
    if not _NUMERIC_CHARS_RE.fullmatch(text):
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(" ", "")
    # '12,5%' -> 12.5: giữ nguyên giá trị phần trăm, việc chia 100 (nếu cần) do query quyết định.
    text = text.rstrip("%")
    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-"):
        negative = not negative
        text = text[1:]

    if "," in text and "." in text:
        # Kiểu VN chuẩn: '.' nghìn, ',' thập phân. Nếu ngược lại (OCR kiểu Anh) thì dấu ',' đứng trước '.'.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        # ',' là dấu nghìn khi mọi nhóm sau dấu đều đúng 3 chữ số (1,234,567).
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
            text = "".join(parts)
        else:
            text = text.replace(",", ".")
    elif "." in text:
        parts = text.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            text = "".join(parts)  # '1.234' / '1.234.567' -> dấu nghìn

    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def is_numeric_cell(value: object) -> bool:
    return parse_vn_number(value) is not None


def answers_match(expected: float, actual: float, *, abs_tol: float = ANSWER_ABS_TOL, rel_tol: float = 0.0) -> bool:
    """Công thức khớp đáp án của cuộc thi: `math.isclose(rel_tol=0.0, abs_tol=1e-2)`."""
    return math.isclose(expected, actual, rel_tol=rel_tol, abs_tol=abs_tol)


def to_float(value: object) -> float | None:
    """Ép kết quả sandbox về float (numpy scalar, Decimal, chuỗi số kiểu VN...)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and (math.isnan(value) or math.isinf(value)) else float(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_float(item())
        except (ValueError, TypeError):
            return None
    if isinstance(value, str):
        return parse_vn_number(value)
    return None
