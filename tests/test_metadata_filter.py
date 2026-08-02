"""Test parse metadata từ câu hỏi + lọc/xếp lại candidate."""

from __future__ import annotations

import pytest

from r2ai.constants import SCOPE_CONSOLIDATED, SCOPE_SEPARATE
from r2ai.extraction.doc_scanner import detect_scope
from r2ai.retrieval.bm25_index import Hit
from r2ai.retrieval.company_meta import CompanyInfo, _aliases
from r2ai.retrieval.metadata_filter import parse_question_meta, parse_years, select_candidates
from r2ai.schemas import TableIndexEntry


@pytest.fixture()
def companies() -> dict[str, CompanyInfo]:
    raw = {
        "VJC": "CTCP Hàng không Vietjet",
        "VNM": "CTCP Sữa Việt Nam",
        "ACB": "Ngân hàng TMCP Á Châu",
        "HPG": "CTCP Tập đoàn Hòa Phát",
        "NKG": "CTCP Thép Nam Kim",
        # Bộ ba thật trong code_stock.csv gây nhầm ticker (xem test bên dưới).
        "FPT": "CTCP FPT",
        "FTS": "CTCP Chứng khoán FPT",
        "FOX": "CTCP Viễn thông FPT",
        "HAG": "CTCP Hoàng Anh Gia Lai",
        "HNG": "CTCP Nông nghiệp Quốc tế Hoàng Anh Gia Lai",
        # Cặp đụng độ substring (không phải nguyên từ): "VIC" nằm trong "Vicem" của tên HT1.
        "VIC": "Tập đoàn VINGROUP - CTCP",
        "HT1": "CTCP Xi Măng Vicem Hà Tiên",
    }
    return {t: CompanyInfo(ticker=t, name=n, aliases=_aliases(n)) for t, n in raw.items()}


def _hit(ticker: str, year: int | None, scope: str | None, score: float, line: int = 1) -> Hit:
    """`line` = số dòng bắt đầu bảng — chính là `<vị trí>` trong table_ref."""
    return Hit(
        entry=TableIndexEntry(
            table_ref=f"{ticker}_financial_statements_{year}_{scope or 'unknown'}|{line}",
            doc_name=f"{ticker}_financial_statements_{year}_{scope or 'unknown'}",
            ticker=ticker,
            year=year,
            scope=scope,
            line=line,
            order=1,
        ),
        score=score,
    )


def test_ticker_in_parens_detected(companies):
    meta = parse_question_meta(
        "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?", companies
    )
    assert meta.ticker_in_parens == {"VJC"}
    assert "VJC" in meta.tickers
    assert meta.years == {2018}
    assert meta.scope == SCOPE_SEPARATE


def test_company_name_without_ticker(companies):
    meta = parse_question_meta("Doanh thu thuần của CTCP Sữa Việt Nam năm 2023 là bao nhiêu?", companies)
    assert meta.tickers == {"VNM"}


def test_bare_ticker_detected(companies):
    meta = parse_question_meta("Tiền của NKG cuối năm 2022 là bao nhiêu tỷ đồng?", companies)
    assert meta.tickers == {"NKG"}
    assert meta.years == {2022}


def test_lowercase_word_not_mistaken_for_ticker(companies):
    # "acb" viết thường trong text thường không phải mã CK -> không được nhận nhầm.
    meta = parse_question_meta("So sánh doanh thu của các công ty acb năm 2020", companies)
    assert "ACB" not in meta.tickers


def test_multi_company_comparison(companies):
    meta = parse_question_meta("Trong năm 2023, so sánh HPG, NKG và VNM về tổng tài sản.", companies)
    assert meta.tickers == {"HPG", "NKG", "VNM"}
    assert meta.scope is None


def test_consolidated_scope_detected(companies):
    meta = parse_question_meta("Tổng tài sản hợp nhất của HPG năm 2021?", companies)
    assert meta.scope == SCOPE_CONSOLIDATED


def test_company_name_containing_other_ticker_does_not_leak(companies):
    """Bug 2: "CTCP Chứng khoán FPT" có mã FTS — chữ "FPT" trong tên không được tính là ticker FPT."""
    meta = parse_question_meta("Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023 là bao nhiêu?", companies)
    assert meta.tickers == {"FTS"}


def test_paren_ticker_wins_over_name_substring(companies):
    meta = parse_question_meta(
        "Tiền gửi ngắn hạn của công ty mẹ Công ty Cổ phần Viễn thông FPT (FOX) cuối năm 2022?", companies
    )
    assert meta.tickers == {"FOX"}
    assert meta.ticker_in_parens == {"FOX"}


def test_real_fpt_question_still_detected(companies):
    """Không được sửa quá tay: câu hỏi về chính Tập đoàn FPT vẫn phải ra FPT."""
    assert parse_question_meta("Giá vốn dịch vụ của FPT trong năm 2024?", companies).tickers == {"FPT"}
    assert parse_question_meta("Tiền của Công ty Cổ phần FPT cuối năm 2022?", companies).tickers == {"FPT"}


def test_two_genuinely_mentioned_companies_both_kept(companies):
    """HAG/HNG: cả 2 công ty đều thật sự được nhắc tới -> không được coi là nhiễu tên."""
    meta = parse_question_meta(
        "Vay dài hạn với Công ty Cổ phần Hoàng Anh Gia Lai của công ty mẹ "
        "CTCP Nông nghiệp Quốc tế Hoàng Anh Gia Lai năm 2017?",
        companies,
    )
    assert meta.tickers == {"HAG", "HNG"}


def test_ticker_inside_another_word_is_not_shadowed(companies):
    """Bug 9: "VIC" chỉ là substring của "Vicem" trong tên HT1 — không được coi là bị che khuất."""
    meta = parse_question_meta("So sánh VIC và CTCP Xi Măng Vicem Hà Tiên về tổng tài sản năm 2023.", companies)
    assert meta.tickers == {"VIC", "HT1"}


def test_shadowing_requires_whole_word_match(companies):
    """Che khuất chỉ áp dụng khi ticker là NGUYÊN MỘT TỪ trong tên công ty kia (FPT), không phải substring."""
    assert parse_question_meta("Tổng tài sản của CTCP Xi Măng Vicem Hà Tiên năm 2023?", companies).tickers == {"HT1"}
    assert parse_question_meta("Lợi nhuận của CTCP Chứng khoán FPT năm 2023?", companies).tickers == {"FTS"}


def test_parse_years_ignores_out_of_range():
    assert parse_years("giai đoạn 2019-2024, so với 1999 và 2099") == {2019, 2024}


def test_detect_scope_from_doc_name():
    assert detect_scope("AAA_financial_statements_2020_separate") == SCOPE_SEPARATE
    assert detect_scope("AAA_financial_statements_2020_consolidated") == SCOPE_CONSOLIDATED
    assert detect_scope("MBS_financial_statements_2021") is None


def test_select_candidates_prefers_metadata_match(companies):
    meta = parse_question_meta("Tiền của NKG cuối năm 2022 là bao nhiêu tỷ đồng?", companies)
    hits = [
        _hit("HPG", 2022, SCOPE_CONSOLIDATED, 30.0, 1),  # điểm cao nhất nhưng sai ticker
        _hit("NKG", 2019, SCOPE_CONSOLIDATED, 20.0, 2),  # đúng ticker, sai năm
        _hit("NKG", 2022, SCOPE_CONSOLIDATED, 10.0, 3),  # khớp đủ
    ]
    selected = select_candidates(hits, meta, top_k=3, fallback_quota=0)
    assert selected[0].entry.ticker == "NKG"
    assert selected[0].entry.year == 2022


def test_select_candidates_keeps_fallback_quota(companies):
    meta = parse_question_meta("Tiền của NKG cuối năm 2022?", companies)
    hits = [_hit("HPG", 2022, SCOPE_CONSOLIDATED, 30.0, i) for i in range(1, 4)]
    hits.append(_hit("NKG", 2022, SCOPE_CONSOLIDATED, 5.0, 9))
    selected = select_candidates(hits, meta, top_k=3, fallback_quota=1)
    tickers = [h.entry.ticker for h in selected]
    assert tickers[0] == "NKG"  # match metadata luôn đứng đầu
    assert "HPG" in tickers  # nhưng vẫn giữ phao fallback


def test_select_candidates_year_expand_allows_next_year(companies):
    meta = parse_question_meta("Tiền của NKG cuối năm 2022?", companies)
    hits = [_hit("NKG", 2023, SCOPE_CONSOLIDATED, 10.0, 1)]
    assert select_candidates(hits, meta, top_k=1, year_expand=True)
    without = select_candidates(hits, meta, top_k=1, year_expand=False)
    assert without and without[0].entry.year == 2023  # vẫn giữ (fallback), nhưng bị xếp sau


def test_select_candidates_empty_and_zero_topk(companies):
    meta = parse_question_meta("Câu hỏi không có công ty nào", companies)
    assert select_candidates([], meta, top_k=5) == []
    assert select_candidates([_hit("HPG", 2020, None, 1.0)], meta, top_k=0) == []


def test_unknown_scope_document_not_filtered_out(companies):
    """55 file không suy ra được scope (MBS/EVF/PRT) không được bị loại oan."""
    meta = parse_question_meta("Tổng tài sản của công ty mẹ HPG năm 2022?", companies)
    hits = [_hit("HPG", 2022, None, 10.0, 1)]
    assert len(select_candidates(hits, meta, top_k=1)) == 1
