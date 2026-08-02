"""Quét `data/financial_statements/` -> danh sách `DocumentRef`.

Layout thật (DATA_DESCRIPTION.md mục 1):
    financial_statements/<TICKER>/<YEAR>/<DOC_NAME>/<DOC_NAME>_extracted.txt

`doc_name` = tên file bỏ `_extracted.txt` — khớp đúng quy ước `relevant_docs` của BTC
(COMPETITION.md mục 3: "tên file cuối cùng trong đường dẫn, loại bỏ phần mở rộng .txt").
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from r2ai.constants import (
    EXTRACTED_SUFFIX,
    MAX_YEAR,
    MIN_YEAR,
    SCOPE_AGGREGATED,
    SCOPE_CONSOLIDATED,
    SCOPE_SEPARATE,
    STATEMENTS_DIR,
)
from r2ai.schemas import DocumentRef

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_YEAR_RE = re.compile(r"(20\d{2})")

_CONSOLIDATED_MARKERS = ("consolidated", "hopnhat", "bctchn")
_SEPARATE_MARKERS = ("separate", "rieng", "congtyme")
_AGGREGATED_MARKERS = ("aggregated",)


def ascii_fold(text: str) -> str:
    """Bỏ dấu + hạ chữ, **giữ nguyên khoảng trắng/dấu phân cách** (còn ranh giới từ)."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D").lower()


def ascii_compact(text: str) -> str:
    """Bỏ dấu + hạ chữ + bỏ ký tự không alnum — dùng cho mọi so khớp lexical."""
    return _NON_ALNUM_RE.sub("", ascii_fold(text))


def ascii_tokens(text: str) -> tuple[str, ...]:
    """Tách thành token alnum đã bỏ dấu — dùng khi cần so khớp **nguyên từ**, không phải substring.

    `ascii_compact("CTCP Xi Măng Vicem Hà Tiên")` = "ctcpximangvicemhatien" chứa "vic" (mã VIC) dù
    thực tế "vicem" là một từ khác hẳn; token hoá tránh được kiểu nhầm đó.
    """
    return tuple(t for t in _NON_ALNUM_RE.split(ascii_fold(text)) if t)


def detect_scope(doc_name: str) -> str | None:
    """Suy ra loại báo cáo từ tên tài liệu; None nếu không xác định được (55 file — mục 3.3)."""
    compact = ascii_compact(doc_name)
    if any(m in compact for m in _AGGREGATED_MARKERS):
        return SCOPE_AGGREGATED
    is_consolidated = any(m in compact for m in _CONSOLIDATED_MARKERS)
    is_separate = any(m in compact for m in _SEPARATE_MARKERS)
    if is_consolidated and not is_separate:
        return SCOPE_CONSOLIDATED
    if is_separate and not is_consolidated:
        return SCOPE_SEPARATE
    return None


def doc_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(EXTRACTED_SUFFIX):
        return name[: -len(EXTRACTED_SUFFIX)]
    return path.stem


def _year_from(path: Path, doc_name: str) -> int | None:
    # Ưu tiên tên thư mục năm (ổn định hơn tên file).
    for part in reversed(path.parts):
        if re.fullmatch(r"20\d{2}", part) and MIN_YEAR <= int(part) <= MAX_YEAR:
            return int(part)
    years = [int(y) for y in _YEAR_RE.findall(doc_name) if MIN_YEAR <= int(y) <= MAX_YEAR]
    return years[-1] if years else None


def scan_documents(statements_dir: Path | None = None) -> list[DocumentRef]:
    """Trả về mọi báo cáo tìm được, sort theo (ticker, year, doc_name) cho deterministic."""
    root = Path(statements_dir) if statements_dir else STATEMENTS_DIR
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy corpus báo cáo: {root}")

    docs: list[DocumentRef] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        doc_name = doc_name_from_path(path)
        if doc_name in seen:
            logger.warning("doc_name trùng %s: %s vs %s (bỏ qua file sau)", doc_name, seen[doc_name], path)
            continue
        seen[doc_name] = path
        try:
            ticker = path.relative_to(root).parts[0]
        except ValueError:  # pragma: no cover - rglob luôn cho path con của root
            ticker = doc_name.split("_")[0]
        docs.append(
            DocumentRef(
                doc_name=doc_name,
                ticker=ticker.upper(),
                year=_year_from(path, doc_name),
                scope=detect_scope(doc_name),
                path=str(path),
            )
        )
    docs.sort(key=lambda d: (d.ticker, d.year or 0, d.doc_name))
    return docs
