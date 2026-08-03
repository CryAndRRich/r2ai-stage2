"""Thực thi `pandas_query` có kiểm soát: AST pre-check + builtins whitelist + timeout.

Không phải sandbox cấp adversarial (không container/seccomp) — không cần, vì code chạy là do
pipeline của mình sinh ra. Nhưng chặt hơn `exec()` trần đáng kể (ARCHITECTURE.md mục 4):
1. AST pre-check: chặn import, gọi eval/exec/compile/open/__import__/getattr..., truy cập attribute `__*`.
2. Namespace builtins whitelist thật (chỉ ~19 hàm), khớp đúng danh sách nêu trong system prompt.
3. Timeout bằng `multiprocessing.Process` (portable, chạy được cả trên Kaggle; `signal.alarm`
   không dùng được trong thread và hành vi lệch giữa các platform).
"""

from __future__ import annotations

import ast
import builtins
import io
import multiprocessing as mp
import queue as queue_module
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

ALLOWED_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    # Tên lớp exception: `except ValueError:` tra tên này lúc runtime, thiếu là NameError ngay
    # trong chính helper parse số mà system prompt yêu cầu LLM viết.
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "ZeroDivisionError",
)

FORBIDDEN_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "exit",
        "quit",
        "memoryview",
    }
)


# Tên attribute bị cấm ở MỌI vị trí trong AST (kể cả chỉ tham chiếu, không gọi) — xem `ast_precheck`.
FORBIDDEN_ATTRS = frozenset({"format", "format_map"})
_DUNDER_IN_TEMPLATE_RE = re.compile(r"\{[^{}]*__")


class SandboxError(Exception):
    """Query không hợp lệ hoặc chạy thất bại."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    ok: bool
    value: object = None
    error: str | None = None
    stdout: str = ""
    # Kết quả khi chạy LẠI cùng code nhưng đọc CSV theo kiểu mặc định `pd.read_csv(path)` —
    # mô phỏng môi trường BTC re-execute (không ai đảm bảo họ dùng `dtype=str`). Chỉ được điền
    # khi gọi với `cross_check_reader=True`; `None` nghĩa là không kiểm.
    alt_value: object = None
    alt_error: str | None = None


def ast_precheck(code: str) -> None:
    """Ném `SandboxError` nếu code chứa cấu trúc bị cấm."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"SyntaxError: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("import bị cấm (pd đã có sẵn trong namespace)")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(f"truy cập attribute nội bộ bị cấm: {node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError(f"truy cập tên nội bộ bị cấm: {node.id}")
        # `str.format`/`format_map` là lỗ hổng riêng: field access `"{0.__class__}"` do mini-parser
        # của chính str xử lý lúc runtime nên KHÔNG sinh ast.Attribute -> vòng kiểm tra `__` ở trên
        # không thấy. Chặn theo **sự tồn tại của tên attribute**, không chỉ khi nó được gọi trực tiếp:
        # `fm = str.format_map` rồi `fm(tmpl, ...)` vẫn phải đi qua một `Attribute(attr="format_map")`.
        # Đây là deny cứng thay vì đuổi theo từng biến thể (alias, nối chuỗi runtime, getattr...).
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise SandboxError(
                f"truy cập `.{node.attr}` bị cấm (str.format có thể lách kiểm tra attribute `__`) — dùng f-string"
            )
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in FORBIDDEN_CALLS:
                raise SandboxError(f"gọi hàm bị cấm: {name}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Lớp phòng thủ phụ (bắt được literal viết thẳng; literal bị chẻ nhỏ rồi ghép runtime
            # thì lớp chặn `.format`/`.format_map` ở trên mới là lớp chặn thật).
            if _DUNDER_IN_TEMPLATE_RE.search(node.value):
                raise SandboxError("chuỗi chứa field template truy cập attribute nội bộ (`{...__...}`)")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxError("global/nonlocal bị cấm")


def _safe_builtins() -> dict[str, object]:
    return {name: getattr(builtins, name) for name in ALLOWED_BUILTINS if hasattr(builtins, name)}


def read_raw_csv(path: str | Path):
    """Đọc CSV đúng như prompt cam kết với LLM: mọi ô là str, không NA, không type inference."""
    import pandas as pd

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        index_col=None,
    )


def read_default_csv(path: str | Path):
    """Đọc CSV theo mặc định `pd.read_csv(path)` — pandas TỰ suy kiểu, cột toàn số thành int/float.

    Dùng để đối chiếu: BTC re-execute `pandas_query` trong môi trường của họ và không có gì đảm bảo
    họ dùng `dtype=str` như sandbox này. Query kiểu `df[df["Mã số"] == "110"]` chạy đúng ở đây nhưng
    filter rỗng ở kia (cột "Mã số" thành int64) -> mất điểm Execution Accuracy mà local không hề báo.
    """
    import pandas as pd

    return pd.read_csv(path, encoding="utf-8-sig", index_col=None)


def _exec_once(code: str, frames: dict, pd_module):
    """Chạy code trên 1 bộ DataFrame, trả về (ok, value|None, error|None, stdout)."""
    import contextlib
    import warnings

    namespace: dict[str, object] = {
        "pd": pd_module,
        "dfs": frames,
        "__builtins__": _safe_builtins(),
        **frames,
    }
    if len(frames) == 1:
        namespace.setdefault("df", next(iter(frames.values())))
    buffer = io.StringIO()
    try:
        # LLM đôi khi dùng API pandas đã deprecate (vd `DataFrame.applymap`) — vẫn chạy đúng, chỉ
        # in FutureWarning/DeprecationWarning gây nhiễu log. Nuốt riêng 2 loại này, KHÔNG nuốt
        # warning khác (vd `RuntimeWarning` chia 0/NaN có thể là dấu hiệu lỗi thật, cần thấy).
        with contextlib.redirect_stdout(buffer), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            exec(compile(code, "<pandas_query>", "exec"), namespace)  # noqa: S102
    except BaseException as exc:  # noqa: BLE001 - lỗi của code sinh ra, không được thoát ra ngoài
        return False, None, f"{type(exc).__name__}: {exc}", buffer.getvalue()[:2000]
    if "result" not in namespace:
        return False, None, "Code không gán biến `result`", buffer.getvalue()[:2000]
    value = namespace["result"]
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalar -> python scalar
        try:
            value = item()
        except (ValueError, TypeError):
            pass
    if not isinstance(value, (int, float, str)):
        # Không pickle DataFrame/Series qua queue (có thể rất lớn) — gửi repr để ghi log lỗi.
        value = repr(value)[:500]
    return True, value, None, buffer.getvalue()[:2000]


def _run(code: str, csv_paths: dict[str, str], queue, cross_check_reader: bool = False) -> None:  # pragma: no cover - chạy ở process con
    """Thân process con: build namespace, exec, đẩy kết quả qua queue.

    Message đầu tiên luôn là `("ready",)` — gửi ngay sau khi `import pandas` xong, để process cha
    bắt đầu bấm giờ timeout từ lúc code người dùng thật sự chạy, không tính thời gian khởi động
    interpreter + import pandas (với `spawn` thì mỗi lần chạy đều phải import lại từ đầu, lúc cache
    OS còn nguội có thể mất hàng chục giây và làm timeout báo nhầm).
    """
    try:
        import pandas as pd

        queue.put(("ready",))
        frames = {var: read_raw_csv(path) for var, path in csv_paths.items()}
        ok, value, error, stdout = _exec_once(code, frames, pd)

        alt_value = alt_error = None
        if cross_check_reader and ok:
            # Cùng 1 process con -> không tốn thêm 1 lần spawn + import pandas (phần đắt nhất).
            try:
                alt_frames = {var: read_default_csv(path) for var, path in csv_paths.items()}
                alt_ok, alt_value, alt_error, _ = _exec_once(code, alt_frames, pd)
                if not alt_ok and alt_error is None:
                    alt_error = "không rõ"
            except BaseException as exc:  # noqa: BLE001 - đối chiếu hỏng không được làm hỏng kết quả chính
                alt_value, alt_error = None, f"{type(exc).__name__}: {exc}"

        queue.put(("done", ok, value, error, stdout, alt_value, alt_error))
    except BaseException as exc:  # noqa: BLE001 - phải bắt hết để không treo process cha
        queue.put(
            ("done", False, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}", "", None, None)
        )


def run_pandas_code(
    code: str,
    csv_paths: dict[str, str | Path],
    *,
    timeout_s: float = 20.0,
    startup_timeout_s: float = 120.0,
    cross_check_reader: bool = False,
) -> ExecutionResult:
    """Chạy `code` với các DataFrame đã load từ `csv_paths` ({variable: đường dẫn CSV}).

    `timeout_s` chỉ tính cho phần code thật sự chạy; `startup_timeout_s` là hạn riêng cho việc
    khởi động process con + `import pandas` (xem `_run`). Trả về `ExecutionResult` thay vì raise,
    để pipeline ghi lại `exec_error` và tiếp tục câu sau.
    """
    if not code.strip():
        return ExecutionResult(ok=False, error="pandas_query rỗng")
    try:
        ast_precheck(code)
    except SandboxError as exc:
        return ExecutionResult(ok=False, error=str(exc))

    missing = [var for var, path in csv_paths.items() if not Path(path).exists()]
    if missing:
        return ExecutionResult(ok=False, error=f"thiếu file CSV cho biến: {', '.join(missing)}")

    # spawn: an toàn trên macOS/Windows và không kế thừa state của process cha.
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_run,
        args=(code, {var: str(path) for var, path in csv_paths.items()}, queue, cross_check_reader),
        daemon=True,
    )
    process.start()
    # Đọc queue TRƯỚC khi join: join trong lúc queue chưa được drain có thể deadlock.
    message = _receive(process, queue, startup_timeout_s, phase="khởi động")
    if isinstance(message, ExecutionResult):
        return message
    if message[0] == "ready":
        message = _receive(process, queue, timeout_s, phase="thực thi")
        if isinstance(message, ExecutionResult):
            return message
    _, ok, value, error, stdout, alt_value, alt_error = message
    _stop(process)
    return ExecutionResult(
        ok=ok, value=value, error=error, stdout=stdout, alt_value=alt_value, alt_error=alt_error
    )


def _receive(process, queue, timeout_s: float, *, phase: str) -> tuple | ExecutionResult:
    """Chờ 1 message từ process con; trả về `ExecutionResult` lỗi nếu timeout/chết bất thường."""
    try:
        return queue.get(timeout=timeout_s)
    except queue_module.Empty:
        alive = process.is_alive()
        _stop(process)
        if alive:
            return ExecutionResult(ok=False, error=f"timeout sau {timeout_s}s ({phase})")
        return ExecutionResult(ok=False, error=f"process con chết bất thường (exitcode={process.exitcode})")


def _stop(process) -> None:
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():  # pragma: no cover
        process.kill()
        process.join()
