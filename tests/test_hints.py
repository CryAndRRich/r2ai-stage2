"""Test gợi ý cột/kỳ/đơn vị nhúng vào prompt (Bug 15).

Mọi case dùng header + câu hỏi **thật** lấy từ dữ liệu cuộc thi (ghi rõ id câu hỏi khi có), vì đây
là lớp thay thế cho việc để model tự đoán — hint sai còn tệ hơn không có hint.
"""

from __future__ import annotations

from r2ai.prompting.hints import (
    AskedPeriod,
    answer_unit,
    asked_period,
    column_hints,
    table_hint_lines,
    table_unit_factor,
    target_column,
    unit_factor,
)


# --- đơn vị của bảng ----------------------------------------------------------------------------


def test_unit_factor_reads_scale_next_to_currency():
    assert unit_factor("31/12/2018VND") == 1.0  # header thật hay dính liền số, không có khoảng trắng
    assert unit_factor("Triệu VND") == 1e6
    assert unit_factor("Số cuối nămTriệu đồng") == 1e6
    assert unit_factor("Đơn vị tính: VND") == 1.0
    assert unit_factor("Đơn vị: nghìn tỷ đồng") == 1e12


def test_unit_factor_ignores_words_that_only_look_like_units_without_diacritics():
    """Bỏ dấu làm "đồng" đụng "động", "tỷ" đụng "ty" (công ty) — đã gặp thật ở cột 'Hoạt động chính'."""
    for name in ("Hoạt động chính", "Biến độngNăm 2018/Năm 2017", "Tên công ty", "Cổ đông lớn", "VNDIRECT"):
        assert unit_factor(name) is None, name


def test_table_unit_factor_prefers_header_then_context():
    assert table_unit_factor(["", "31/12/2016Triệu VND", "31/12/2015Triệu VND"]) == 1e6
    assert table_unit_factor(["", "Số cuối năm", "Số đầu năm"], "Đơn vị tính: VND") == 1.0
    # Câu văn ngữ cảnh nói về số của CHÍNH nó, không phải đơn vị bảng -> không suy bừa.
    assert table_unit_factor(["", "Số cuối năm"], "chi phí lãi vay được vốn hóa là 31.729 triệu VND") is None


# --- đơn vị đáp án của câu hỏi ------------------------------------------------------------------


def test_answer_unit_basic_scales():
    assert answer_unit("Lãi tiền gửi năm 2018 của VJC là bao nhiêu triệu đồng?").factor == 1e6
    assert answer_unit("Tổng tài sản cuối năm 2020 là bao nhiêu tỷ đồng?").factor == 1e9
    assert answer_unit("Tổng tài sản là bao nhiêu nghìn tỷ đồng?").factor == 1e12


def test_answer_unit_handles_compound_scale_tram_ty():
    """66/1.012 câu hỏi dùng "trăm tỷ đồng" — hiểu thành "tỷ đồng" là sai hệ số 100 lần."""
    assert answer_unit("Lợi nhuận thuần sau thuế năm 2021 là bao nhiêu trăm tỷ đồng?").factor == 1e11
    assert answer_unit("Lãi thuần hoạt động tài chính năm 2017 là mấy trăm tỷ đồng?").factor == 1e11
    assert answer_unit("Tính tổng giá trị cổ tức theo đơn vị trăm tỷ đồng?").factor == 1e11


def test_answer_unit_marks_ratio_questions():
    unit = answer_unit("Tỷ lệ nợ trên vốn chủ sở hữu của HPG năm 2022 là bao nhiêu phần trăm?")
    assert unit.no_conversion and unit.factor is None
    assert answer_unit("Hệ số thanh toán hiện hành năm 2021 là bao nhiêu lần?").no_conversion


def test_answer_unit_ignores_unit_stated_before_the_question_word():
    """"Vốn điều lệ 5.000 tỷ đồng ... chiếm bao nhiêu phần trăm?" -> tỉ lệ, KHÔNG phải tỷ đồng."""
    unit = answer_unit("Vốn điều lệ 5.000 tỷ đồng của công ty chiếm bao nhiêu phần trăm?")
    assert unit.no_conversion and unit.factor is None


def test_answer_unit_non_monetary_answers():
    """"Năm nào ... cao nhất" trả về một NĂM; đổi đơn vị tiền ở đây là phá đáp án."""
    assert answer_unit("Năm nào có tổng nợ phải trả của HND cao nhất trong các năm 2016, 2017?").no_conversion
    assert answer_unit("Có bao nhiêu doanh nghiệp có doanh thu tăng?").no_conversion
    unit = answer_unit("Chênh lệch số lượng cổ phiếu cuối năm 2018 là bao nhiêu triệu cổ phiếu?")
    assert unit.no_conversion and "1e+06" in unit.reason


def test_answer_unit_ratio_words_and_percent_before_question_word():
    """Câu hỏi thật: "Tốc độ tăng trưởng % ... là bao nhiêu?" và "Tính tỷ số nợ ..." đều là tỉ lệ."""
    assert answer_unit("Tốc độ tăng trưởng % tổng tiền của VIC từ 2019 đến 2021 là bao nhiêu?").no_conversion
    assert answer_unit("Tính tỷ số nợ ngắn hạn trên vốn chủ sở hữu của DNH năm 2025.").no_conversion
    # nhưng có đơn vị tiền sau từ hỏi thì tiền thắng, `%` chỉ là dữ kiện
    assert answer_unit("Công ty chiếm 5% thị phần có doanh thu là bao nhiêu tỷ đồng?").factor == 1e9


def test_answer_unit_foreign_currency_is_not_converted():
    unit = answer_unit("Số dư ngoại tệ USD của ACV vào cuối năm 2018 là bao nhiêu triệu USD?")
    assert unit.no_conversion and "ngoại tệ" in unit.reason


def test_answer_unit_reads_bare_scale_without_currency_word():
    """Khẩu ngữ: "... nợ nhiều hơn ... mấy triệu thế nhỉ?" — vẫn là triệu đồng."""
    assert answer_unit("Tính đến cuối năm 2022 thì BIDV nợ nhiều hơn MSB mấy triệu thế nhỉ?").factor == 1e6


def test_answer_unit_counts_shares_and_banks_as_non_monetary():
    assert answer_unit("Tổng số lượng cổ phần của BVH vào cuối năm 2018 là bao nhiêu cổ phần?").no_conversion
    assert answer_unit("Bao nhiêu ngân hàng thuộc nhóm NVB, SGB có chênh lệch dương?").no_conversion


def test_answer_unit_defaults_to_raw_vnd_when_question_states_no_unit():
    """Ví dụ submission của BTC: "Doanh thu thuần ... năm 2023 là bao nhiêu?" -> 63075000000 (VND thô)."""
    unit = answer_unit("Doanh thu thuần của Công ty CP Sữa Việt Nam (VNM) năm 2023 là bao nhiêu?")
    assert unit.factor == 1.0 and unit.assumed


# --- kỳ của từng cột ----------------------------------------------------------------------------


def test_column_hints_reads_dates_years_and_relative_labels():
    hints = column_hints(["", "31/12/2018 | Triệu đồng", "31/12/2017 | %", "Năm trước"])
    assert [h.period for h in hints] == [None, "31/12/2018", "31/12/2017", "năm trước"]
    assert [h.year for h in hints] == [None, 2018, 2017, None]
    assert hints[1].unit_factor == 1e6
    assert hints[2].is_percent and hints[2].unit_factor is None


def test_target_column_picks_asked_date():
    """id=65 thật: "đến ngày 31/12/2016" — V1 lấy cột 31/12/2015."""
    hints = column_hints(["", "31/12/2016Triệu VND", "31/12/2015Triệu VND"])
    assert target_column(hints, asked_period("... đến ngày 31/12/2016 là bao nhiêu triệu đồng?")) == 1


def test_target_column_picks_asked_year():
    """id=1 thật: "năm 2018" — V1 lấy cột 2017."""
    hints = column_hints(["", "2018VND", "2017VND"])
    assert target_column(hints, asked_period("Lãi tiền gửi năm 2018 ... ?", [2018])) == 1


def test_target_column_uses_relative_labels_with_report_year():
    hints = column_hints(["", "Số cuối năm", "Số đầu năm"])
    asked_end = asked_period("Trả trước cuối năm 2021 là bao nhiêu?", [2021])
    assert target_column(hints, asked_end, table_year=2021) == 1
    asked_start = asked_period("Thuế TNDN đầu năm 2017 là bao nhiêu triệu đồng?", [2017])
    assert target_column(hints, asked_start, table_year=2017) == 2
    # "cuối năm 2016" = "đầu năm 2017": report 2017 vẫn trả lời được.
    asked_prev = asked_period("... cuối năm 2016 ...", [2016])
    assert target_column(hints, asked_prev, table_year=2017) == 2


def test_target_column_maps_start_of_year_to_end_of_previous_year():
    """Bảng ghi ngày cụ thể thường không có cột 01/01/N — "đầu năm 2018" nằm ở cột 31/12/2017."""
    hints = column_hints(["", "31/12/2018VND", "31/12/2017VND"])
    assert target_column(hints, asked_period("Chi phí trả trước đầu năm 2018 là bao nhiêu?", [2018])) == 2
    # và ngược lại: "cuối năm 2017" nằm ở cột 01/01/2018 của report 2018.
    hints2 = column_hints(["", "31/12/2018VND", "01/01/2018VND"])
    assert target_column(hints2, asked_period("Tiền cuối năm 2017 là bao nhiêu?", [2017])) == 2


def test_target_column_separates_balance_labels_from_fiscal_year_labels():
    """Case thật id=154/124: "Tiền ... đầu năm 2022" trên báo cáo lưu chuyển tiền tệ.

    Ở đó "đầu năm" là NHÃN DÒNG, cột vẫn là niên độ được hỏi ("Năm nay") — không phải "Năm trước".
    Ngược lại, bảng số dư có cột "Số đầu năm" thật thì phải lấy đúng cột đó.
    """
    cashflow = column_hints(["Mã số", "CHỈ TIÊU", "Thuyết minh", "Năm nay", "Năm trước"])
    asked = asked_period("Tiền và tương đương tiền đầu năm 2022 của NLG là bao nhiêu tỷ đồng?", [2022])
    assert target_column(cashflow, asked, table_year=2022) == 3

    balance = column_hints(["", "Số cuối nămTriệu đồng", "Số đầu nămTriệu đồng"])
    assert target_column(balance, asked_period("... đầu năm 2017 ...", [2017]), table_year=2017) == 2

    # Niên độ N nằm ở cột "Năm trước" của report N+1.
    assert target_column(cashflow, asked_period("Chi phí lãi vay năm 2022 ...", [2022]), table_year=2023) == 4


def test_target_column_prefers_main_value_column_over_provision():
    """Cùng kỳ nhưng nhiều cột: "Dự phòng"/"Hao mòn lũy kế" không phải cột giá trị chính."""
    hints = column_hints(
        ["", "Số cuối năm | Dự phòng | VND", "Số cuối năm | Giá gốc | VND", "Số đầu năm | Giá gốc | VND"]
    )
    asked = asked_period("Giá gốc hàng tồn kho cuối năm 2022 là bao nhiêu?", [2022])
    assert target_column(hints, asked, table_year=2022) == 2


def test_target_column_returns_none_when_header_has_no_period():
    hints = column_hints(["", "Quyền sử dụng đất triệu đồng", "Phần mềm máy vi tính triệu đồng"])
    assert target_column(hints, AskedPeriod(year=2021, wants_end=True), table_year=2021) is None


def test_asked_period_detects_start_and_end():
    assert asked_period("Thuế TNDN ... đầu năm 2017 ...").wants_start
    assert asked_period("Tổng tài sản đến ngày 31/12/2020 ...").wants_end
    assert asked_period("... cuối năm 2019 ...").wants_end


# --- gói chú thích cho prompt -------------------------------------------------------------------


def test_table_hint_lines_gives_column_and_scale():
    hint, target = table_hint_lines(
        ["", "31/12/2018VND", "1/1/2018VND"],
        "29. Doanh thu hoạt động tài chính",
        question="Lãi tiền gửi năm 2018 của VJC là bao nhiêu triệu đồng?",
        years=[2018],
        table_year=2018,
    )
    assert target == 1
    assert ".iloc[:, 1]" in hint
    assert "÷ 1.000.000" in hint


def test_table_hint_lines_says_no_conversion_when_units_already_match():
    hint, target = table_hint_lines(
        ["", "31/12/2016Triệu VND", "31/12/2015Triệu VND"],
        "",
        question="Dư nợ cho vay ... đến ngày 31/12/2016 là bao nhiêu triệu đồng?",
        years=[2016],
        table_year=2016,
    )
    assert target == 1
    assert "KHÔNG cần" in hint


def test_table_hint_lines_uses_report_year_for_multi_year_questions():
    """Câu so sánh nhiều năm: với bảng của report 2021 thì cột cần lấy là kỳ 2021, không phải năm cuối câu."""
    hint, target = table_hint_lines(
        ["", "31/12/2021 | VND", "01/01/2021 | VND"],
        "",
        question="Năm nào có doanh thu cao nhất trong các năm 2020, 2021, 2022 và 2025?",
        years=[2020, 2021, 2022, 2025],
        table_year=2021,
    )
    assert target == 1
    assert "KHÔNG đổi" in hint  # "năm nào" -> đáp án là một năm


def test_table_hint_lines_admits_when_it_cannot_tell():
    hint, target = table_hint_lines(
        ["", "Giá trị ghi sổ tại ngày mua VND", "Điều chỉnh giá trị hợp lý VND"],
        "",
        question="Tỷ trọng tài sản cố định vô hình trên tổng tài sản đến ngày 31/12/2016 (%)?",
        years=[2016],
        table_year=2016,
    )
    assert target is None
    assert "chưa xác định được" in hint


def test_percent_column_is_excluded_when_question_asks_money():
    hints = column_hints(["", "31/12/2018 | Triệu đồng", "31/12/2018 | %"])
    _hint, target = table_hint_lines(
        ["", "31/12/2018 | Triệu đồng", "31/12/2018 | %"],
        "",
        question="Cho vay các tổ chức kinh tế đến ngày 31/12/2018 là bao nhiêu triệu đồng?",
        years=[2018],
        table_year=2018,
    )
    assert hints[2].is_percent
    assert target == 1  # không được trả về cột "%"
