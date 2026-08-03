"""Test sandbox (chặn code không hợp lệ, timeout) + parse số kiểu Việt Nam."""

from __future__ import annotations

import pytest

import queue as queue_module

from r2ai.execution.numeric import answers_match, is_numeric_cell, parse_vn_number, to_float
from r2ai.execution.sandbox import SandboxError, _run, ast_precheck, run_pandas_code


@pytest.fixture()
def csv_file(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text(
        "Mã số,Chỉ tiêu,Số cuối năm\n"
        "110,Tiền và các khoản tương đương tiền,816.523.338.816\n"
        "120,Đầu tư tài chính ngắn hạn,301.600.000.000\n",
        encoding="utf-8",
    )
    return path


# --- AST pre-check ---------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = 1",
        "from pathlib import Path\nresult = 1",
        "result = eval('1+1')",
        "result = exec('x=1')",
        "result = open('/etc/passwd').read()",
        "result = __import__('os').getcwd()",
        "result = df.__class__.__mro__[0]",
        "result = getattr(df, 'shape')[0]",
        "result = globals()",
    ],
)
def test_ast_precheck_blocks_forbidden(code):
    with pytest.raises(SandboxError):
        ast_precheck(code)


@pytest.mark.parametrize(
    "code",
    [
        'result = "{0.__class__}".format(df1)',
        'result = "{0.__class__.__mro__[1]}".format(df1)',
        'result = "{}".format_map({})',
        'x = "{0.__dict__}"\nresult = 1',
        # Bypass tìm được ở vòng review thứ 2: alias tên hàm + ghép chuỗi `__` lúc runtime.
        'fm = str.format_map\nparts = ["{a.", "_", "_class__}"]\nresult = fm("".join(parts), {"a": df1})',
        'f = str.format\nresult = f("{0.__class__}", df1)',
        # Chỉ tham chiếu, chưa gọi -> vẫn phải chặn (không đợi tới lúc gọi mới biết).
        'g = "{}".format\nresult = 1.0',
    ],
)
def test_ast_precheck_blocks_str_format_bypass(code):
    """Bug 6: field access của str.format không sinh ast.Attribute nên vòng kiểm tra `__` không thấy."""
    with pytest.raises(SandboxError):
        ast_precheck(code)


def test_format_bypass_blocked_end_to_end(csv_file):
    """Chạy thật qua sandbox: bypass vòng 2 không còn leak được repr của object nội bộ."""
    code = (
        'fm = str.format_map\n'
        'parts = ["{a.", "_", "_class__}"]\n'
        'tmpl = "".join(parts)\n'
        'result = fm(tmpl, {"a": df1})\n'
    )
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=60)
    assert not outcome.ok
    assert "format_map" in (outcome.error or "")


def test_fstring_still_allowed(csv_file):
    """Chặn `.format` không được làm mất cách format hợp lệ mà prompt khuyến nghị (f-string)."""
    outcome = run_pandas_code("x = 3.14159\nresult = float(f'{x:.2f}')", {"df1": csv_file}, timeout_s=60)
    assert outcome.ok and outcome.value == 3.14


def test_ast_precheck_allows_normal_pandas():
    ast_precheck("s = df1['Số cuối năm'].str.replace('.', '', regex=False)\nresult = float(s.iloc[0])")


def test_ast_precheck_allows_prompt_number_helper():
    """Helper `to_num` được `finalize_code()` ghép vào code phải qua được pre-check."""
    from r2ai.execution.numeric import TO_NUM_HELPER_SOURCE

    ast_precheck(TO_NUM_HELPER_SOURCE + "\nresult = to_num('1.234')")


def test_prompt_number_helper_matches_numeric_module():
    """Bug 1: `TO_NUM_HELPER_SOURCE` (ghép vào pandas_query bởi `finalize_code`) phải cho cùng
    kết quả với `numeric.parse_vn_number` — 2 bản không được lệch nhau."""
    from r2ai.execution.numeric import TO_NUM_HELPER_SOURCE

    namespace: dict = {}
    exec(TO_NUM_HELPER_SOURCE, namespace)  # noqa: S102 - chạy chính đoạn sẽ ghép vào pandas_query
    to_num = namespace["to_num"]
    for raw in [
        "1.234.567",
        "816.523.338.816",
        "12,5",
        "12,5%",
        "(1.234)",
        "(12,5)",
        "-1.234",
        "1.234,56",
        "1,234,567",  # kiểu Anh — công thức cũ trong prompt raise ValueError ở đây
        "1,234,567.89",
        "2020",
        "0",
        "",
        "Thuyết minh",
    ]:
        assert to_num(raw) == parse_vn_number(raw), raw


def test_exception_names_available_in_sandbox(csv_file):
    """`except ValueError:` trong helper của LLM phải tra được tên lớp exception."""
    code = "try:\n    v = float('x')\nexcept ValueError:\n    v = 1.0\nresult = v"
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=60)
    assert outcome.ok and outcome.value == 1.0


# --- Đối chiếu 2 cách đọc CSV (bảo vệ Execution Accuracy khi BTC re-execute) -------


@pytest.fixture()
def code_table(tmp_path):
    """Bảng có cột "Mã số" toàn số — pandas mặc định sẽ suy thành int64, `dtype=str` thì không."""
    path = tmp_path / "coded.csv"
    path.write_text(
        "Chỉ tiêu,Mã số,Số cuối năm\nTiền,110,1.234.567\nĐầu tư,120,2.000\n", encoding="utf-8"
    )
    return path


def test_cross_check_flags_query_depending_on_dtype_str(code_table):
    """`df["Mã số"] == "110"` chạy đúng với dtype=str nhưng filter rỗng khi pandas suy kiểu."""
    code = "row = df1[df1['Mã số'] == '110']\nresult = float(len(row))"
    outcome = run_pandas_code(code, {"df1": code_table}, timeout_s=60, cross_check_reader=True)
    assert outcome.ok and outcome.value == 1.0  # với dtype=str: khớp 1 dòng
    assert outcome.alt_value == 0.0  # với reader mặc định: không khớp dòng nào -> lệch, phải phát hiện được


def test_cross_check_passes_for_astype_str_query(code_table):
    """Cùng ý nghĩa nhưng viết bền (`.astype(str)`) thì 2 cách đọc CSV cho cùng kết quả."""
    code = "row = df1[df1['Mã số'].astype(str).str.strip() == '110']\nresult = float(len(row))"
    outcome = run_pandas_code(code, {"df1": code_table}, timeout_s=60, cross_check_reader=True)
    assert outcome.ok and outcome.value == 1.0
    assert outcome.alt_value == 1.0 and outcome.alt_error is None


def test_cross_check_reports_error_from_alt_reader(code_table):
    """Query dùng `.str` trên cột số: reader mặc định raise AttributeError -> phải ghi lại alt_error."""
    code = "result = float(len(df1[df1['Mã số'].str.contains('110', na=False)]))"
    outcome = run_pandas_code(code, {"df1": code_table}, timeout_s=60, cross_check_reader=True)
    assert outcome.ok and outcome.value == 1.0
    assert outcome.alt_error is not None and "AttributeError" in outcome.alt_error


def test_cross_check_off_by_default(code_table):
    outcome = run_pandas_code("result = float(len(df1))", {"df1": code_table}, timeout_s=60)
    assert outcome.ok and outcome.alt_value is None and outcome.alt_error is None


def test_to_num_helper_is_loader_agnostic(code_table):
    """to_num phải cho cùng kết quả dù ô là str hay đã bị pandas suy thành số."""
    from r2ai.prompting.build_prompt import finalize_code

    code = finalize_code("result = to_num(df1.iloc[1, 2])")  # "2.000" (str) vs 2000 (int64)
    outcome = run_pandas_code(code, {"df1": code_table}, timeout_s=60, cross_check_reader=True)
    assert outcome.ok and outcome.value == 2000.0
    assert outcome.alt_value == 2000.0, outcome.alt_error


def test_to_num_callable_as_series_method(csv_file):
    """Pilot thật: LLM gọi `col.to_num()` như method -> AttributeError, mất trắng câu.
    Helper ghép vào pandas_query giờ gắn luôn method cho Series/DataFrame nên cả 2 cách đều chạy."""
    from r2ai.prompting.build_prompt import finalize_code

    code = finalize_code("result = float(df1.iloc[:, 2].to_num().sum())")
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=60, cross_check_reader=True)
    assert outcome.ok, outcome.error
    assert outcome.value == pytest.approx(816.523338816e9 + 301.6e9, rel=1e-9)
    assert outcome.alt_value == outcome.value  # bền với cả cách đọc CSV mặc định


def test_to_num_dataframe_method(csv_file):
    from r2ai.prompting.build_prompt import finalize_code

    code = finalize_code("nums = df1.iloc[:, 2:].to_num()\nresult = float(nums.max().max())")
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=60)
    assert outcome.ok and outcome.value == pytest.approx(816523338816.0)


def test_ast_precheck_reports_syntax_error():
    with pytest.raises(SandboxError, match="SyntaxError"):
        ast_precheck("result = (1 +")


# --- Thực thi --------------------------------------------------------------------


def test_run_pandas_code_happy_path(csv_file):
    code = (
        "row = df1[df1.iloc[:, 0] == '110']\n"
        "raw = row.iloc[0, 2]\n"
        "result = round(float(raw.replace('.', '')) / 1e9, 2)\n"
    )
    outcome = run_pandas_code(code, {"df1": csv_file}, timeout_s=60)
    assert outcome.ok, outcome.error
    assert outcome.value == pytest.approx(816.52, abs=0.01)


def test_run_pandas_code_dfs_dict_available(csv_file):
    outcome = run_pandas_code("result = len(dfs['df1'])", {"df1": csv_file}, timeout_s=60)
    assert outcome.ok and outcome.value == 2


def test_run_pandas_code_cells_are_strings(csv_file):
    """Prompt cam kết dtype=str, keep_default_na=False — sandbox phải giữ đúng cam kết đó."""
    outcome = run_pandas_code("result = int(isinstance(df1.iloc[0, 2], str))", {"df1": csv_file}, timeout_s=60)
    assert outcome.ok and outcome.value == 1


def test_run_pandas_code_requires_result(csv_file):
    outcome = run_pandas_code("x = 1", {"df1": csv_file}, timeout_s=60)
    assert not outcome.ok and "result" in (outcome.error or "")


def test_run_pandas_code_blocks_import(csv_file):
    outcome = run_pandas_code("import os\nresult = 1", {"df1": csv_file}, timeout_s=60)
    assert not outcome.ok and "import" in (outcome.error or "")


def test_run_pandas_code_builtins_whitelist(csv_file):
    outcome = run_pandas_code("result = len(sorted([3, 1, 2]))", {"df1": csv_file}, timeout_s=60)
    assert outcome.ok and outcome.value == 3
    blocked = run_pandas_code("result = hash('x')", {"df1": csv_file}, timeout_s=60)
    assert not blocked.ok  # `hash` không nằm trong whitelist -> NameError


def test_run_pandas_code_timeout(csv_file):
    outcome = run_pandas_code("while True:\n    pass\nresult = 1", {"df1": csv_file}, timeout_s=2)
    assert not outcome.ok and "timeout" in (outcome.error or "")


def test_run_pandas_code_missing_csv(tmp_path):
    outcome = run_pandas_code("result = 1", {"df1": tmp_path / "nope.csv"}, timeout_s=10)
    assert not outcome.ok and "CSV" in (outcome.error or "")


def test_run_pandas_code_empty_query(csv_file):
    assert not run_pandas_code("   ", {"df1": csv_file}).ok


def test_deprecated_pandas_api_warning_is_suppressed(csv_file, recwarn):
    """LLM đôi khi dùng API pandas đã deprecate (vd `.applymap`) — vẫn phải chạy đúng và KHÔNG in
    FutureWarning/DeprecationWarning ra ngoài (chỉ là noise, không phải lỗi thật cần thấy).

    Gọi trực tiếp `_run` (thân thực thi, không qua multiprocessing) để `recwarn` của pytest bắt
    được warning trong CÙNG process — nếu code suppress warning bên trong `_run` không hoạt động,
    warning sẽ lọt ra tới đây và test fail.
    """
    q: queue_module.Queue = queue_module.Queue()
    _run(
        "result = float(df1.applymap(lambda x: x).shape[0])",
        {"df1": str(csv_file)},
        q,
    )
    assert q.get_nowait() == ("ready",)
    msg = q.get_nowait()
    assert msg[1] is True, msg  # (kind, ok, value, error, stdout)
    assert msg[2] == 2.0
    assert not any(issubclass(w.category, (FutureWarning, DeprecationWarning)) for w in recwarn.list), [
        str(w.message) for w in recwarn.list
    ]


def test_run_pandas_code_non_scalar_result_is_reported(csv_file):
    outcome = run_pandas_code("result = df1", {"df1": csv_file}, timeout_s=60)
    assert outcome.ok  # exec chạy được...
    assert to_float(outcome.value) is None  # ...nhưng không phải số -> pipeline đánh dấu fail


# --- Số kiểu Việt Nam ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234.567", 1234567.0),
        ("816.523.338.816", 816523338816.0),
        ("12,5", 12.5),
        ("12,5%", 12.5),
        ("(1.234)", -1234.0),
        ("(12,5)", -12.5),
        ("-1.234", -1234.0),
        ("1.234,56", 1234.56),
        ("1,234,567.89", 1234567.89),  # OCR đôi khi ra kiểu Anh
        ("2020", 2020.0),
        ("0", 0.0),
    ],
)
def test_parse_vn_number(raw, expected):
    assert parse_vn_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", "Tiền và các khoản tương đương", "Thuyết minh", None, "n/a"])
def test_parse_vn_number_rejects_non_numeric(raw):
    assert parse_vn_number(raw) is None


def test_is_numeric_cell():
    assert is_numeric_cell("1.234")
    assert not is_numeric_cell("TÀI SẢN NGẮN HẠN")


def test_answers_match_uses_competition_tolerance():
    assert answers_match(100.0, 100.005)
    assert not answers_match(100.0, 100.5)


def test_to_float_handles_numpy_and_bool():
    import numpy as np

    assert to_float(np.float64(3.5)) == 3.5
    assert to_float(True) is None
    assert to_float("1.234") == 1234.0
    assert to_float(float("nan")) is None
