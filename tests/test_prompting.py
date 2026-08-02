"""Test dựng prompt + hậu xử lý output LLM (bỏ fence/import, nhận diện biến đã dùng)."""

from __future__ import annotations

import pytest

from r2ai.prompting.build_prompt import build_prompt, extract_code, used_variables, variable_name
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


def test_used_variables_detects_only_referenced():
    variables = {"df1": "a|1", "df2": "b|2"}
    assert used_variables("result = float(df2.iloc[0, 1])", variables) == ["df2"]


def test_used_variables_treats_dfs_as_all():
    variables = {"df1": "a|1", "df2": "b|2"}
    assert used_variables("result = len(dfs)", variables) == ["df1", "df2"]


def test_used_variables_no_partial_match():
    assert used_variables("result = df10", {"df1": "a|1"}) == []
