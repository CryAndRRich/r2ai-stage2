"""Tokenizer tiếng Việt cho BM25: NFC normalize + casefold + word segmentation.

`underthesea.word_tokenize` là mặc định (khớp thiết kế baseline ViFinQA). Nếu chưa cài
underthesea, tự động fallback sang tách theo ranh giới ký tự Unicode — kém hơn nhưng vẫn
chạy được, và log cảnh báo **một lần** để không im lặng đổi hành vi.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_word_tokenize = None
_fallback_warned = False


def _get_word_tokenize():
    global _word_tokenize
    if _word_tokenize is None:
        try:
            from underthesea import word_tokenize  # type: ignore[import-not-found]

            _word_tokenize = word_tokenize
        except Exception:  # ImportError hoặc lỗi tải model
            _word_tokenize = False
    return _word_tokenize


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace(" ", " ").casefold()
    return _WS_RE.sub(" ", normalized).strip()


def tokenize(text: str) -> list[str]:
    """Trả về list token đã normalize (dùng cho cả corpus lúc index và query lúc search)."""
    global _fallback_warned
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokenizer = _get_word_tokenize()
    if tokenizer:
        try:
            return [t for t in tokenizer(normalized) if t.strip()]
        except Exception as exc:  # pragma: no cover - lỗi runtime của underthesea
            logger.warning("underthesea lỗi (%s) -> dùng tokenizer fallback", exc)
    elif not _fallback_warned:
        logger.warning("Chưa cài underthesea -> dùng tokenizer regex fallback (chất lượng BM25 thấp hơn)")
        _fallback_warned = True
    return _TOKEN_RE.findall(normalized)


def using_underthesea() -> bool:
    """Cho phép ghi vào manifest index biết corpus đã tokenize bằng tokenizer nào."""
    return bool(_get_word_tokenize())
