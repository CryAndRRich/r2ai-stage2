"""Dựng prompt sinh `pandas_query` từ 1 `RetrievalResult`.

Quy ước tên biến: bảng thứ i nhúng vào prompt được gán `df1`, `df2`, ... — đúng ràng buộc
`evidence[].variable` của COMPETITION.md (identifier Python hợp lệ, không trùng trong 1 câu hỏi).
Ánh xạ variable -> table_ref được trả về cùng prompt để bước sau dựng `evidence` chính xác.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from r2ai.constants import TEMPLATES_DIR
from r2ai.execution.numeric import TO_NUM_HELPER_SOURCE
from r2ai.schemas import RetrievalCandidate, RetrievalResult

SYSTEM_TEMPLATE = "system_pandas.txt"
USER_TEMPLATE = "user_template.txt"


@dataclass(frozen=True, slots=True)
class PandasPrompt:
    system: str
    user: str
    variables: dict[str, str]  # variable -> table_ref
    candidates: tuple[RetrievalCandidate, ...]


@lru_cache(maxsize=8)
def _template(name: str, templates_dir_str: str) -> str:
    return (Path(templates_dir_str) / name).read_text(encoding="utf-8")


def variable_name(position: int) -> str:
    """1-indexed -> df1, df2, ..."""
    return f"df{position}"


def _metadata_block(result: RetrievalResult) -> str:
    lines = [f"- Câu hỏi id: {result.id}"]
    lines.append(f"- Mã CK nhận diện được: {', '.join(result.tickers) if result.tickers else 'không rõ'}")
    lines.append(
        f"- Năm được hỏi: {', '.join(str(y) for y in result.years) if result.years else 'không rõ'}"
    )
    scope_text = {
        "separate": "báo cáo riêng / công ty mẹ",
        "consolidated": "báo cáo hợp nhất",
        "aggregated": "báo cáo tổng hợp",
    }.get(result.scope or "", "không nêu rõ")
    lines.append(f"- Loại báo cáo được hỏi: {scope_text}")
    return "\n".join(lines)


def _table_block(variable: str, candidate: RetrievalCandidate) -> str:
    scope = candidate.scope or "không rõ"
    context = candidate.context_before.strip() or "(không có)"
    header = (
        f'<table variable="{variable}" table_ref="{candidate.table_ref}" '
        f'ticker="{candidate.ticker}" year="{candidate.year}" scope="{scope}" page="{candidate.page}">'
    )
    return f"{header}\n<!-- Ngữ cảnh ngay trước bảng: {context} -->\n{candidate.csv_text.rstrip()}\n</table>"


def build_prompt(
    result: RetrievalResult,
    *,
    max_tables: int = 4,
    max_csv_chars: int | None = None,
    max_prompt_chars: int | None = None,
    templates_dir: Path | None = None,
) -> PandasPrompt:
    """Nhúng tối đa `max_tables` candidate đầu vào prompt (thứ tự đã xếp ở bước retrieval).

    Nếu prompt vượt `max_prompt_chars`, bỏ dần bảng ở cuối (candidate xếp sau = ít liên quan hơn)
    cho tới khi vừa — thà mất 1 bảng phụ còn hơn bị cắt ngang giữa CSV.
    """
    templates = str(templates_dir or TEMPLATES_DIR)
    system = _template(SYSTEM_TEMPLATE, templates).strip()
    user_template = _template(USER_TEMPLATE, templates)

    selected = list(result.candidates[:max_tables])
    while True:
        candidates: list[RetrievalCandidate] = []
        for candidate in selected:
            csv_text = candidate.csv_text
            if max_csv_chars is not None and len(csv_text) > max_csv_chars:
                cut = csv_text[:max_csv_chars]
                newline = cut.rfind("\n")
                csv_text = (cut[:newline] if newline > 0 else cut) + "\n"
            candidates.append(candidate.model_copy(update={"csv_text": csv_text}))

        variables = {variable_name(i): c.table_ref for i, c in enumerate(candidates, start=1)}
        tables_block = "\n\n".join(
            _table_block(variable_name(i), c) for i, c in enumerate(candidates, start=1)
        )
        user = (
            user_template.replace("{{QUESTION}}", result.question)
            .replace("{{QUESTION_METADATA}}", _metadata_block(result))
            .replace("{{TABLES}}", tables_block or "(không có bảng nào được truy hồi)")
        ).strip()

        if max_prompt_chars is None or len(system) + len(user) <= max_prompt_chars or len(selected) <= 1:
            return PandasPrompt(
                system=system, user=user, variables=variables, candidates=tuple(candidates)
            )
        selected = selected[:-1]


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_MAX_TRIM_LINES = 60


def _parse(code: str) -> ast.Module | None:
    if not code.strip():
        return None
    try:
        return ast.parse(code, mode="exec")
    except SyntaxError:
        return None


def _assigns_result(tree: ast.Module) -> bool:
    """Có gán vào biến `result` không — dấu hiệu chắc chắn nhất rằng đây là code chứ không phải văn xuôi."""
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "result" for t in targets):
            return True
    return False


def _trim_to_parsable(text: str) -> str:
    """Cắt câu dẫn/lời giải thích bám quanh code cho tới khi phần còn lại là code chạy được.

    Model chat vẫn hay trả về "Here is the code:\\nresult = ..." **không kèm markdown fence** dù
    system prompt đã cấm giải thích; nếu giữ nguyên thì `ast.parse` raise SyntaxError và mất trắng
    một câu trả lời lẽ ra đúng. Dùng chính `ast.parse` làm bộ nhận diện "đâu là code" thay vì đoán
    bằng regex, và **ưu tiên đoạn có gán `result`** — một dòng văn xuôi chỉ gồm 1 từ (ví dụ
    "Kết quả") cũng parse được thành công như một biểu thức Name, nên "parse được" chưa đủ.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    tree = _parse(stripped)
    if tree is not None and _assigns_result(tree):
        return stripped

    lines = stripped.splitlines()[:_MAX_TRIM_LINES]
    fallback: str | None = stripped if tree is not None else None
    for start in range(len(lines)):
        if not lines[start].strip():
            continue
        for end in range(len(lines), start, -1):
            candidate = "\n".join(lines[start:end]).strip()
            parsed = _parse(candidate)
            if parsed is None:
                continue
            if _assigns_result(parsed):
                return candidate
            if fallback is None:
                fallback = candidate
    return fallback if fallback is not None else stripped


def extract_code(completion: str) -> str:
    """Lấy code từ output LLM: bỏ markdown fence, bỏ dòng import, bỏ lời dẫn ngoài code.

    Model instruct vẫn hay bọc fence dù prompt cấm — xử lý ở đây thay vì kỳ vọng model tuân thủ 100%.
    """
    text = completion.strip()
    fences = _CODE_FENCE_RE.findall(text)
    if fences:
        text = max(fences, key=len)
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and " " in stripped:
            continue  # runtime đã có `pd`; import bị sandbox chặn
        if stripped.startswith(("```", "#!")):
            continue
        lines.append(line)
    return _trim_to_parsable("\n".join(lines).strip())


def finalize_code(code: str) -> str:
    """Ghép hàm `to_num()` chuẩn vào TRƯỚC code LLM sinh ra (nối chuỗi văn bản, không phải inject
    vào namespace sandbox) — để `pandas_query` cuối cùng vẫn tự chứa 100% mà LLM không phải tốn
    ~400 token chép lại nguyên hàm ở mỗi câu (system_pandas.txt chỉ mô tả hành vi, không bắt viết).
    Nếu LLM vẫn lỡ tự định nghĩa `to_num` riêng (bỏ qua hướng dẫn mới) thì không sao — hàm của nó
    định nghĩa sau sẽ đè lên, code vẫn chạy đúng, chỉ tốn dư token chứ không hỏng.
    """
    if not code.strip():
        return code
    return f"{TO_NUM_HELPER_SOURCE}\n\n{code}"


def used_variables(code: str, variables: dict[str, str]) -> list[str]:
    """Các biến DataFrame thực sự xuất hiện trong code (giữ thứ tự df1, df2, ...).

    Dùng để dựng `evidence` chỉ gồm bảng thật sự được query — nếu liệt kê cả bảng không dùng thì
    `evidence` sẽ chứa CSV vô ích và làm nặng gói nộp bài.
    """
    used = [name for name in variables if re.search(rf"\b{re.escape(name)}\b", code)]
    if re.search(r"\bdfs\b", code):  # dùng dict dfs -> coi như dùng tất cả
        return list(variables)
    return used
