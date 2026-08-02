"""Parse metadata từ câu hỏi (ticker / năm / loại báo cáo) và lọc-xếp lại candidate BM25.

Toàn bộ là rule lexical, không dùng model — rẻ, deterministic, và kiểm chứng được miễn phí
nhờ 229 câu hỏi (22,6%) có sẵn mã CK trong ngoặc (xem `r2ai/sanity/retrieval_probe.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from r2ai.constants import MAX_YEAR, MIN_YEAR, SCOPE_CONSOLIDATED, SCOPE_SEPARATE
from r2ai.extraction.doc_scanner import ascii_compact, ascii_tokens
from r2ai.retrieval.bm25_index import Hit
from r2ai.retrieval.company_meta import CompanyInfo

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# Dấu hiệu câu hỏi nhắm vào báo cáo riêng lẻ (công ty mẹ) hoặc hợp nhất.
_SEPARATE_MARKERS = ("congtyme", "baocaorieng", "bctcrieng", "rienglẻ", "rienngle", "baocaotaichinhrieng")
_CONSOLIDATED_MARKERS = ("hopnhat", "baocaohopnhat", "bctchn", "toantapdoan")


@dataclass(frozen=True, slots=True)
class QuestionMeta:
    tickers: frozenset[str] = frozenset()
    years: frozenset[int] = frozenset()
    scope: str | None = None  # separate | consolidated | None (không nêu rõ)
    ticker_in_parens: frozenset[str] = frozenset()  # ticker viết trong ngoặc — dùng để sanity check


def _tickers_in_parens(question: str, known: set[str]) -> set[str]:
    found: set[str] = set()
    for group in re.findall(r"\(([^)]{1,40})\)", question):
        for token in re.findall(r"[A-Z0-9]{3,4}", group.upper()):
            if token in known:
                found.add(token)
    return found


def _tickers_mentioned(question: str, known: set[str]) -> set[str]:
    """Khớp ticker viết hoa đứng riêng (VNM, VJC...). Chỉ nhận UPPERCASE để giảm false positive.

    Ví dụ cần tránh: chữ "SO" trong "so sánh" viết hoa ở đầu câu; đó là lý do dùng regex biên
    ký tự alnum chứ không phải substring, và chỉ so trên bản gốc (không hạ chữ).
    """
    upper = question.upper()
    found: set[str] = set()
    for ticker in known:
        if re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", upper) and ticker in question:
            found.add(ticker)
    return found


def _tickers_by_company_name(question: str, companies: dict[str, CompanyInfo]) -> set[str]:
    compact = ascii_compact(question)
    return {
        ticker
        for ticker, info in companies.items()
        if any(alias and alias in compact for alias in info.aliases)
    }


def parse_years(question: str) -> set[int]:
    return {int(y) for y in _YEAR_RE.findall(question) if MIN_YEAR <= int(y) <= MAX_YEAR}


def parse_scope(question: str) -> str | None:
    compact = ascii_compact(question)
    is_separate = any(m in compact for m in _SEPARATE_MARKERS)
    is_consolidated = any(m in compact for m in _CONSOLIDATED_MARKERS)
    if is_separate and not is_consolidated:
        return SCOPE_SEPARATE
    if is_consolidated and not is_separate:
        return SCOPE_CONSOLIDATED
    return None


def _shadowed_tickers(
    mentioned: set[str], by_name: set[str], in_parens: set[str], companies: dict[str, CompanyInfo]
) -> set[str]:
    """Ticker "trần" thật ra chỉ là một phần trong tên đầy đủ của công ty KHÁC đã match.

    Ví dụ thật trong `questions.jsonl`: "CTCP Chứng khoán FPT" có mã **FTS**, nhưng chữ "FPT"
    trong tên lại trùng mã của Tập đoàn FPT; tương tự "Công ty Cổ phần Viễn thông FPT (FOX)".
    Nếu không loại, candidate của FPT sẽ chiếm chỗ trong ngân sách top_k rất hẹp -> tụt precision.

    Chỉ loại khi ticker đó KHÔNG tự xuất hiện trong ngoặc và KHÔNG phải mã của một công ty được
    match theo tên — tức là bằng chứng duy nhất cho nó chính là chuỗi nằm trong tên công ty khác.

    So khớp theo **nguyên từ** (token), không phải substring: `ascii_compact` bỏ hết khoảng trắng
    nên "vic" (mã VIC) là substring của "…vicem…" trong tên HT1 (CTCP Xi Măng Vicem Hà Tiên) và
    sẽ loại nhầm VIC ở câu hỏi nhắc cả hai công ty. Quét toàn bộ 100×99 cặp trong `code_stock.csv`
    thì VIC/HT1 là cặp đụng độ substring duy nhất ngoài họ FPT/FTS/FOX.
    """
    shadowed: set[str] = set()
    for ticker in mentioned - in_parens - by_name:
        needle = ascii_compact(ticker)
        if any(
            needle in _name_tokens(companies[other].name) for other in by_name if other != ticker
        ):
            shadowed.add(ticker)
    return shadowed


@lru_cache(maxsize=512)
def _name_tokens(name: str) -> frozenset[str]:
    return frozenset(ascii_tokens(name))


def parse_question_meta(question: str, companies: dict[str, CompanyInfo]) -> QuestionMeta:
    known = set(companies)
    in_parens = _tickers_in_parens(question, known)
    mentioned = _tickers_mentioned(question, known)
    by_name = _tickers_by_company_name(question, companies)
    # Union: câu so sánh nhiều công ty (14,2%) thường liệt kê ticker trần, không có tên đầy đủ.
    tickers = (mentioned - _shadowed_tickers(mentioned, by_name, in_parens, companies)) | by_name | in_parens
    return QuestionMeta(
        tickers=frozenset(tickers),
        years=frozenset(parse_years(question)),
        scope=parse_scope(question),
        ticker_in_parens=frozenset(in_parens),
    )


@dataclass(slots=True)
class RankedHit:
    hit: Hit
    rank: int
    ticker_ok: bool
    year_ok: bool
    scope_ok: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def matches_all(self) -> bool:
        return self.ticker_ok and self.year_ok and self.scope_ok


def _year_ok(hit_year: int | None, years: frozenset[int], *, expand: bool) -> bool:
    if not years:
        return True
    if hit_year is None:
        return True  # không suy ra được năm -> không loại oan
    if hit_year in years:
        return True
    # "cuối năm N" có thể nằm ở cột "Số đầu năm" của report N+1.
    return expand and any(hit_year == y + 1 for y in years)


def _scope_ok(hit_scope: str | None, scope: str | None) -> bool:
    if scope is None or hit_scope is None:
        return True
    return hit_scope == scope


def annotate(hits: list[Hit], meta: QuestionMeta, *, year_expand: bool) -> list[RankedHit]:
    ranked: list[RankedHit] = []
    for rank, hit in enumerate(hits):
        entry = hit.entry
        ticker_ok = not meta.tickers or entry.ticker in meta.tickers
        year_ok = _year_ok(entry.year, meta.years, expand=year_expand)
        scope_ok = _scope_ok(entry.scope, meta.scope)
        reasons: list[str] = []
        if not ticker_ok:
            reasons.append("ticker_mismatch")
        if not year_ok:
            reasons.append("year_mismatch")
        if not scope_ok:
            reasons.append("scope_mismatch")
        ranked.append(
            RankedHit(hit=hit, rank=rank, ticker_ok=ticker_ok, year_ok=year_ok, scope_ok=scope_ok, reasons=reasons)
        )
    return ranked


def select_candidates(
    hits: list[Hit],
    meta: QuestionMeta,
    *,
    top_k: int,
    fallback_quota: int = 0,
    year_expand: bool = True,
) -> list[Hit]:
    """Ưu tiên hit khớp đủ metadata, sau đó khớp một phần, cuối cùng giữ `fallback_quota` hit trần.

    Giữ lại quota fallback là chủ ý: nếu parse ticker/năm sai (OCR, tên công ty lạ) thì việc lọc
    cứng sẽ cho recall 0; quota nhỏ này là phao cứu sinh mà gần như không tốn precision.
    """
    if top_k <= 0:
        return []
    ranked = annotate(hits, meta, year_expand=year_expand)

    exact: list[RankedHit] = []
    partial: list[RankedHit] = []
    rest: list[RankedHit] = []
    for item in ranked:
        if item.matches_all:
            exact.append(item)
        elif item.ticker_ok and (item.year_ok or item.scope_ok):
            partial.append(item)
        else:
            rest.append(item)

    # Trong nhóm exact: ưu tiên năm khớp chính xác (không phải năm mở rộng N+1).
    def exact_key(item: RankedHit) -> tuple[int, int]:
        precise_year = 0 if (not meta.years or item.hit.entry.year in meta.years) else 1
        return precise_year, item.rank

    exact.sort(key=exact_key)
    partial.sort(key=lambda r: r.rank)
    rest.sort(key=lambda r: r.rank)

    has_constraint = bool(meta.tickers or meta.years or meta.scope)
    quota = min(fallback_quota, max(0, top_k - 1)) if has_constraint else 0
    primary = [*exact, *partial]
    selected = [r.hit for r in primary[: top_k - quota]]
    selected.extend(r.hit for r in rest[:quota])

    if len(selected) < top_k:
        chosen = {h.entry.table_ref for h in selected}
        for r in (*primary, *rest):
            if len(selected) >= top_k:
                break
            if r.hit.entry.table_ref not in chosen:
                selected.append(r.hit)
                chosen.add(r.hit.entry.table_ref)
    return selected[:top_k]
