"""Validate cấu trúc submission theo COMPETITION.md mục 3 rồi zip.

    python -m r2ai.packaging.zip_submission            # validate + zip
    python -m r2ai.packaging.zip_submission --check-only

Kiểm tra đúng từng quy định của BTC:
- Chỉ 1 file `.json` trong zip; `submission.json` và `data/` nằm ngay gốc zip (không có thư mục cha).
- Mọi `csv_path` là đường dẫn tương đối bắt đầu bằng `data/` và file phải tồn tại.
- `variable` là identifier Python hợp lệ và không trùng trong cùng 1 câu hỏi.
- Đủ id (mặc định 1..1012), không thiếu, không trùng.
- `answer` là số thực; `relevant_tables` đúng dạng `<doc>|<int>`; `relevant_docs` khớp phần doc.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from pathlib import Path

from r2ai.constants import QUESTIONS_PATH, SUBMISSION_BUILD_DIR, SUBMISSION_ZIP_PATH

logger = logging.getLogger(__name__)

_TABLE_REF_RE = re.compile(r"^.+\|\d+$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# File báo cáo nội bộ do bước assemble ghi ra cùng thư mục build — KHÔNG được nén vào zip và
# không tính vào quy định "zip chỉ chứa 1 file .json".
INTERNAL_JSON_FILES = frozenset({"assemble_stats.json", "reexec_report.json"})

REQUIRED_FIELDS = (
    "id",
    "question",
    "answer",
    "relevant_docs",
    "relevant_tables",
    "evidence",
    "pandas_query",
)


def expected_ids(questions_path: Path) -> set[int]:
    path = Path(questions_path)
    if not path.exists():
        return set()
    ids: set[int] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(int(json.loads(line)["id"]))
    return ids


def validate_build(build_dir: Path, *, questions_path: Path | None = None) -> list[str]:
    """Trả về danh sách lỗi (rỗng = hợp lệ).

    `questions_path=None` -> bỏ qua phần kiểm tra "đủ id": dùng cho self-check cấu trúc bên trong
    `zip_submission()`, nơi không nhất thiết biết file câu hỏi. CLI luôn truyền đường dẫn thật nên
    vẫn kiểm tra đủ 1.012 câu như quy định của BTC.
    """
    errors: list[str] = []
    submission_path = build_dir / "submission.json"
    data_dir = build_dir / "data"

    json_files = sorted(p for p in build_dir.glob("*.json") if p.name not in INTERNAL_JSON_FILES)
    if not submission_path.exists():
        errors.append(f"thiếu {submission_path}")
        return errors
    if len(json_files) > 1:
        errors.append(f"zip chỉ được chứa 1 file .json, đang có: {[p.name for p in json_files]}")
    if not data_dir.is_dir():
        errors.append(f"thiếu thư mục {data_dir}")

    try:
        items = json.loads(submission_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"submission.json không parse được: {exc}")
        return errors
    if not isinstance(items, list):
        errors.append("submission.json phải là một JSON array")
        return errors

    seen_ids: set[int] = set()
    referenced_csv: set[str] = set()
    for pos, item in enumerate(items):
        tag = f"item[{pos}]"
        if not isinstance(item, dict):
            errors.append(f"{tag} không phải object")
            continue
        for field in REQUIRED_FIELDS:
            if field not in item:
                errors.append(f"{tag} thiếu field `{field}`")
        qid = item.get("id")
        if not isinstance(qid, int) or isinstance(qid, bool):
            errors.append(f"{tag} `id` phải là integer, nhận {qid!r}")
        else:
            tag = f"id={qid}"
            if qid in seen_ids:
                errors.append(f"{tag} trùng id")
            seen_ids.add(qid)
        if not isinstance(item.get("question"), str) or not item.get("question"):
            errors.append(f"{tag} `question` phải là string không rỗng")
        answer = item.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (int, float)):
            errors.append(f"{tag} `answer` phải là số, nhận {answer!r}")
        if not isinstance(item.get("pandas_query"), str):
            errors.append(f"{tag} `pandas_query` phải là string")

        tables = item.get("relevant_tables")
        if not isinstance(tables, list) or not all(isinstance(t, str) for t in tables):
            errors.append(f"{tag} `relevant_tables` phải là list[str]")
            tables = []
        for ref in tables:
            if not _TABLE_REF_RE.match(ref):
                errors.append(f"{tag} table_ref sai định dạng `<doc>|<int>`: {ref!r}")

        docs = item.get("relevant_docs")
        if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
            errors.append(f"{tag} `relevant_docs` phải là list[str]")
            docs = []
        table_docs = {ref.rsplit("|", 1)[0] for ref in tables if "|" in ref}
        missing_docs = table_docs - set(docs)
        if missing_docs:
            errors.append(f"{tag} `relevant_docs` thiếu doc của relevant_tables: {sorted(missing_docs)}")

        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{tag} `evidence` phải là list")
            continue
        variables: set[str] = set()
        for ev in evidence:
            if not isinstance(ev, dict):
                errors.append(f"{tag} evidence item không phải object")
                continue
            variable = ev.get("variable")
            csv_path = ev.get("csv_path")
            if not isinstance(variable, str) or not _IDENTIFIER_RE.match(variable or ""):
                errors.append(f"{tag} `variable` không phải identifier Python hợp lệ: {variable!r}")
            elif variable in variables:
                errors.append(f"{tag} `variable` trùng trong cùng câu hỏi: {variable}")
            else:
                variables.add(variable)
            if not isinstance(csv_path, str) or not csv_path.startswith("data/"):
                errors.append(f"{tag} `csv_path` phải là đường dẫn tương đối bắt đầu bằng `data/`: {csv_path!r}")
                continue
            if Path(csv_path).is_absolute() or ".." in Path(csv_path).parts:
                errors.append(f"{tag} `csv_path` không được tuyệt đối hoặc chứa `..`: {csv_path}")
                continue
            referenced_csv.add(csv_path)
            if not (build_dir / csv_path).exists():
                errors.append(f"{tag} thiếu file CSV được tham chiếu: {csv_path}")

    wanted = expected_ids(questions_path) if questions_path is not None else set()
    if wanted:
        missing = wanted - seen_ids
        extra = seen_ids - wanted
        if missing:
            errors.append(f"thiếu {len(missing)} câu hỏi (ví dụ: {sorted(missing)[:10]})")
        if extra:
            errors.append(f"có {len(extra)} id không thuộc questions.jsonl (ví dụ: {sorted(extra)[:10]})")

    if data_dir.is_dir():
        orphans = {f"data/{p.name}" for p in data_dir.glob("*.csv")} - referenced_csv
        if orphans:
            logger.warning("%d file CSV trong data/ không được evidence nào tham chiếu", len(orphans))
    return errors


class SubmissionInvalidError(Exception):
    """Cấu trúc submission không hợp lệ — không nén."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__(f"{len(errors)} lỗi validate: " + "; ".join(errors[:5]))
        self.errors = errors


def zip_submission(
    build_dir: Path, zip_path: Path, *, questions_path: Path | None = None, validate: bool = True
) -> Path:
    """Nén đúng cấu trúc: `submission.json` + `data/*.csv` ở gốc zip.

    Tự validate trước khi nén (`validate=True`) để hàm an toàn kể cả khi bị gọi thẳng từ notebook,
    không đi qua `main()`. Truyền `questions_path` nếu muốn kiểm tra luôn phần "đủ id".
    """
    if validate:
        errors = validate_build(build_dir, questions_path=questions_path)
        if errors:
            raise SubmissionInvalidError(errors)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(build_dir / "submission.json", "submission.json")
        for csv_file in sorted((build_dir / "data").glob("*.csv")):
            zf.write(csv_file, f"data/{csv_file.name}")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate + zip submission")
    parser.add_argument("--build-dir", type=Path, default=SUBMISSION_BUILD_DIR)
    parser.add_argument("--zip-path", type=Path, default=SUBMISSION_ZIP_PATH)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    errors = validate_build(args.build_dir, questions_path=args.questions)
    if errors:
        print(f"❌ {len(errors)} lỗi validate:")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... và {len(errors) - 50} lỗi nữa")
        return 1
    print("✅ Cấu trúc submission hợp lệ")
    if args.check_only:
        return 0
    # validate=False: vừa validate ngay phía trên với đúng questions_path, không cần chạy lại.
    path = zip_submission(args.build_dir, args.zip_path, questions_path=args.questions, validate=False)
    size_mb = path.stat().st_size / 1e6
    print(f"✅ Đã tạo {path} ({size_mb:.1f} MB) — upload thủ công tại leaderboard.aiguru.com.vn")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
