"""Test dựng prompt + hậu xử lý output LLM (bỏ fence/import, nhận diện biến đã dùng)."""

from __future__ import annotations

import re

import pytest

from r2ai.constants import TEMPLATES_DIR
from r2ai.execution.numeric import TO_NUM_HELPER_SOURCE, parse_vn_number
from r2ai.execution.sandbox import run_pandas_code
from r2ai.prompting.build_prompt import build_prompt, extract_code, finalize_code, used_variables, variable_name
from r2ai.schemas import RetrievalCandidate, RetrievalResult


def _result(n: int = 3, csv_text: str = "Chỉ tiêu,Số cuối năm\nTiền,1.000\n") -> RetrievalResult:
    return RetrievalResult(
        id=1,
        question="Tiền của NKG cuối năm 2022 là bao nhiêu tỷ đồng?",
        tickers=["NKG"],
        years=[2022],
        scope="consolidated",
        candidates=[
            RetrievalCandidate(
                table_ref=f"NKG_financial_statements_2022_consolidated|{i}",
                doc_name="NKG_financial_statements_2022_consolidated",
                ticker="NKG",
                year=2022,
                scope="consolidated",
                page=10 + i,
                score=10.0 - i,
                rank=i - 1,
                context_before="BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT. Đơn vị: VND",
                csv_text=csv_text,
            )
            for i in range(1, n + 1)
        ],
    )


def test_variable_names_are_1_indexed():
    assert variable_name(1) == "df1"
    assert variable_name(3) == "df3"


def test_build_prompt_embeds_tables_and_variables():
    prompt = build_prompt(_result(3), max_tables=2)
    assert list(prompt.variables) == ["df1", "df2"]
    assert 'variable="df1"' in prompt.user
    assert "NKG_financial_statements_2022_consolidated|1" in prompt.user
    assert "Đơn vị: VND" in prompt.user
    assert "df3" not in prompt.user
    assert "2022" in prompt.user and "NKG" in prompt.user


def test_build_prompt_truncates_csv_at_line_boundary():
    prompt = build_prompt(_result(1, csv_text="a,b\n" + "1,2\n" * 100), max_tables=1, max_csv_chars=20)
    body = prompt.candidates[0].csv_text
    assert len(body) <= 21 and body.endswith("\n")


def test_build_prompt_drops_tables_when_over_budget():
    big = "a,b\n" + "1,2\n" * 500
    prompt = build_prompt(_result(3, csv_text=big), max_tables=3, max_prompt_chars=3000)
    assert len(prompt.variables) < 3


def test_build_prompt_keeps_at_least_one_table_even_if_over_budget():
    big = "a,b\n" + "1,2\n" * 500
    prompt = build_prompt(_result(3, csv_text=big), max_tables=3, max_prompt_chars=10)
    assert len(prompt.variables) == 1


def test_build_prompt_without_candidates():
    result = _result(1).model_copy(update={"candidates": []})
    prompt = build_prompt(result)
    assert prompt.variables == {}
    assert "không có bảng" in prompt.user


def test_build_prompt_embeds_precomputed_column_and_unit_hints():
    """Bug 15: prompt phải nói rõ cột nào là kỳ được hỏi + hệ số đổi đơn vị, không để model tự đoán."""
    csv_text = ",31/12/2022VND,01/01/2022VND\nTiền,1.000.000.000,900.000.000\n"
    prompt = build_prompt(_result(1, csv_text=csv_text), max_tables=1)
    assert "Đơn vị đáp án: tỷ đồng" in prompt.user
    assert "CỘT ỨNG VỚI KỲ ĐƯỢC HỎI -> .iloc[:, 1]" in prompt.user
    assert "÷ 1.000.000.000" in prompt.user


def test_build_prompt_hint_admits_uncertainty_instead_of_guessing():
    csv_text = "Chỉ tiêu,Tổng cộng\nTiền,1.000\n"
    prompt = build_prompt(_result(1, csv_text=csv_text), max_tables=1)
    assert "chưa xác định được" in prompt.user


def test_system_prompt_sample_pattern_runs_and_picks_the_hinted_column(tmp_path):
    """Đoạn mẫu trong `system_pandas.txt` phải CHẠY THẬT và lấy đúng cột kỳ được hỏi.

    Case thật id=1 (VJC): bảng có cột 2018 và 2017, câu hỏi hỏi 2018 theo triệu đồng.
    Mẫu cũ (`found[-1]`) trả 69.917.578.051 (cột 2017, không đổi đơn vị) — sai cả cột lẫn đơn vị.
    """
    sample = re.search(r"```python\n(.*?)```", _system_template(), re.DOTALL)
    assert sample is not None, "system_pandas.txt phải còn đoạn mẫu code"
    code = finalize_code(sample.group(1).replace("<nhãn dòng>", "Lãi tiền gửi"))

    csv_file = tmp_path / "table.csv"
    csv_file.write_text(",2018VND,2017VND\nLãi tiền gửi,208.253.201.298,69.917.578.051\n", encoding="utf-8")
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=15, cross_check_reader=True)
    assert outcome.ok
    # 208.253.201.298 / 1e6, KHÔNG phải cột 2017 — và giữ nguyên độ chính xác (ngưỡng chấm là
    # 0,02% TƯƠNG ĐỐI nên không được làm tròn về 2 chữ số thập phân).
    assert outcome.value == 208253.201298
    assert outcome.alt_value == outcome.value  # bền với cả `pd.read_csv` mặc định (Bug 13)


def test_system_prompt_sample_pattern_survives_missing_label(tmp_path):
    sample = re.search(r"```python\n(.*?)```", _system_template(), re.DOTALL)
    code = finalize_code(sample.group(1).replace("<nhãn dòng>", "Nhãn không tồn tại"))
    csv_file = tmp_path / "table.csv"
    csv_file.write_text(",2018VND\nLãi tiền gửi,208.253.201.298\n", encoding="utf-8")
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=15)
    assert outcome.ok and outcome.value == 0.0  # không crash, không gán None


def test_system_prompt_forbids_taking_the_last_numeric_value():
    template = _system_template()
    assert "vals[-1]" in template and "NEVER take" in template


def test_system_prompt_matches_competition_tolerance_and_percent_convention():
    """BTC (discussion): sai số ≤ 0,02% TƯƠNG ĐỐI, và đáp án % trả về theo thang % (90, không phải 0,9)."""
    template = _system_template()
    assert "0.02% RELATIVE" in template
    assert "Do NOT round" in template
    assert "round(" not in template  # đoạn mẫu không còn làm tròn 2 chữ số thập phân
    assert "`result = 90.0` is correct" in template


def _system_template() -> str:
    return (TEMPLATES_DIR / "system_pandas.txt").read_text(encoding="utf-8")


def test_extract_code_strips_fence_and_imports():
    completion = "Đây là code:\n```python\nimport pandas as pd\nresult = float(df1.iloc[0, 1])\n```\n"
    assert extract_code(completion) == "result = float(df1.iloc[0, 1])"


def test_extract_code_without_fence():
    assert extract_code("result = 1.0\n") == "result = 1.0"


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("Here is the code:\nresult = float(df1.iloc[0, 1])", "result = float(df1.iloc[0, 1])"),
        ("Sure! The code:\nresult = 1.0\nThis returns the value in VND.", "result = 1.0"),
        ("Kết quả\nresult = 2.0\nHết.", "result = 2.0"),
        (
            "Để tính, lấy dòng đầu:\nrow = df1.iloc[0]\nresult = float(row[1])",
            "row = df1.iloc[0]\nresult = float(row[1])",
        ),
    ],
)
def test_extract_code_strips_prose_without_fence(completion, expected):
    """Bug 4: model chat vẫn hay kèm câu dẫn mà không bọc markdown fence."""
    assert extract_code(completion) == expected


def test_extract_code_keeps_multiline_code_with_helper():
    completion = "def to_num(s):\n    return float(s)\nresult = to_num('1')"
    assert extract_code(completion) == completion


def test_extract_code_gives_up_gracefully_on_pure_prose():
    assert extract_code("Xin lỗi, tôi không thể trả lời.") == "Xin lỗi, tôi không thể trả lời."
    assert extract_code("") == ""


def test_finalize_code_prepends_helper_and_stays_runnable(tmp_path):
    """LLM không còn phải tự viết `to_num` (Bug 1 tối ưu tốc độ) — `finalize_code` ghép vào,
    kết quả vẫn là code tự chứa chạy đúng qua sandbox thật."""
    code = finalize_code("result = to_num(df1.iloc[0, 1])")
    assert code.startswith("def to_num(s):")
    assert code.rstrip().endswith("result = to_num(df1.iloc[0, 1])")

    csv_file = tmp_path / "table.csv"
    csv_file.write_text("Chỉ tiêu,Số cuối năm\nTiền,1.234.567\n", encoding="utf-8")
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=10)
    assert outcome.ok and outcome.value == 1234567.0


def test_finalize_code_noop_on_empty():
    assert finalize_code("") == ""
    assert finalize_code("   ") == "   "


def test_finalize_code_still_correct_if_llm_redefines_helper_anyway():
    """Nếu LLM bỏ qua hướng dẫn mới và tự định nghĩa `to_num` riêng, định nghĩa của nó (đứng sau)
    đè lên bản chuẩn — không hỏng, chỉ dư token."""
    code = finalize_code("def to_num(s):\n    return 999.0\nresult = to_num('1.234')")
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102
    assert namespace["result"] == 999.0


def test_to_num_helper_source_matches_parse_vn_number():
    namespace: dict = {}
    exec(TO_NUM_HELPER_SOURCE, namespace)  # noqa: S102
    to_num = namespace["to_num"]
    for raw in ["1.234.567", "12,5", "12,5%", "(1.234)", "1.234,56", "1,234,567", "", "Thuyết minh"]:
        assert to_num(raw) == parse_vn_number(raw), raw


def test_used_variables_detects_only_referenced():
    variables = {"df1": "a|1", "df2": "b|2"}
    assert used_variables("result = float(df2.iloc[0, 1])", variables) == ["df2"]


def test_used_variables_treats_dfs_as_all():
    variables = {"df1": "a|1", "df2": "b|2"}
    assert used_variables("result = len(dfs)", variables) == ["df1", "df2"]


def test_used_variables_no_partial_match():
    assert used_variables("result = df10", {"df1": "a|1"}) == []
