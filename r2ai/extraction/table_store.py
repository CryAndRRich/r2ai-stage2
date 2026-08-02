"""Cache extraction theo từng report + đọc lazy CSV của một bảng.

Vì sao tách cache theo document: extraction toàn bộ 1.973 file là bước tốn CPU nhất, và tổng
CSV của ~150K bảng quá lớn để giữ hết trong RAM hay nhúng hết vào `tables_index.jsonl`.
Do đó:
- `tables_cache/<doc_name>.json` — grid đầy đủ của mọi bảng trong report, kèm fingerprint file nguồn.
- `tables_index.jsonl` — chỉ text ngắn để BM25 index.
CSV đầy đủ chỉ đọc khi cần (top-k retrieval, đóng gói submission).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from r2ai.constants import EXTRACTION_VERSION, TABLES_CACHE_DIR
from r2ai.extraction.html_tables import grid_to_csv, split_header
from r2ai.schemas import DocumentRef, TableAsset, TableIndexEntry


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    size: int
    mtime_ns: int

    def as_dict(self) -> dict[str, int]:
        return {"size": self.size, "mtime_ns": self.mtime_ns}


def fingerprint(path: Path) -> SourceFingerprint:
    stat = path.stat()
    return SourceFingerprint(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def cache_path(doc_name: str, cache_dir: Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir else TABLES_CACHE_DIR
    return root / f"{doc_name}.json"


def write_cache(
    doc: DocumentRef,
    grids: list[list[list[str]]],
    metas: list[dict[str, Any]],
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Ghi cache của 1 report. `metas[i]` gồm `line`, `order`, `page`, `context_before`, `closed`."""
    path = cache_path(doc.doc_name, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "extraction_version": EXTRACTION_VERSION,
        "fingerprint": fingerprint(Path(doc.path)).as_dict(),
        "doc": doc.model_dump(),
        "tables": [
            {
                "line": meta["line"],
                "order": meta.get("order", 0),
                "page": meta.get("page"),
                "context_before": meta.get("context_before", ""),
                "closed": meta.get("closed", True),
                "grid": grid,
            }
            for grid, meta in zip(grids, metas, strict=True)
        ],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_cache(doc_name: str, *, cache_dir: Path | None = None) -> dict[str, Any] | None:
    path = cache_path(doc_name, cache_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("extraction_version") != EXTRACTION_VERSION:
        return None
    return payload


def is_cache_fresh(doc: DocumentRef, *, cache_dir: Path | None = None) -> bool:
    payload = read_cache(doc.doc_name, cache_dir=cache_dir)
    if payload is None:
        return False
    try:
        current = fingerprint(Path(doc.path)).as_dict()
    except OSError:
        return False
    return payload.get("fingerprint") == current


@lru_cache(maxsize=32)
def _cached_doc(doc_name: str, cache_dir_str: str) -> dict[str, Any] | None:
    return read_cache(doc_name, cache_dir=Path(cache_dir_str))


def load_table(doc_name: str, line: int, *, cache_dir: Path | None = None) -> TableAsset | None:
    """Đọc 1 bảng từ cache theo **số dòng bắt đầu** (khoá của `table_ref`).

    LRU giữ vài report gần nhất để retrieval theo cụm không phải đọc lại đĩa.
    """
    root = str(Path(cache_dir) if cache_dir else TABLES_CACHE_DIR)
    payload = _cached_doc(doc_name, root)
    if payload is None:
        return None
    doc = payload["doc"]
    for entry in payload["tables"]:
        if entry["line"] != line:
            continue
        header, rows = split_header(entry["grid"])
        return TableAsset(
            doc_name=doc["doc_name"],
            ticker=doc["ticker"],
            year=doc.get("year"),
            scope=doc.get("scope"),
            line=line,
            order=entry.get("order", 0),
            page=entry.get("page"),
            context_before=entry.get("context_before", ""),
            header=header,
            rows=rows,
        )
    return None


def table_to_csv(table: TableAsset, *, max_chars: int | None = None) -> str:
    """CSV text của 1 bảng; cắt theo biên dòng nếu vượt `max_chars` (không cắt giữa dòng)."""
    csv_text = grid_to_csv([table.header, *table.rows] if table.header else table.rows)
    if max_chars is not None and len(csv_text) > max_chars:
        cut = csv_text[:max_chars]
        newline = cut.rfind("\n")
        csv_text = (cut[:newline] if newline > 0 else cut) + "\n"
    return csv_text


def load_table_csv(
    doc_name: str, line: int, *, cache_dir: Path | None = None, max_chars: int | None = None
) -> str:
    table = load_table(doc_name, line, cache_dir=cache_dir)
    if table is None:
        return ""
    return table_to_csv(table, max_chars=max_chars)


def load_table_csv_by_ref(
    table_ref: str, *, cache_dir: Path | None = None, max_chars: int | None = None
) -> str:
    doc_name, line = parse_table_ref(table_ref)
    return load_table_csv(doc_name, line, cache_dir=cache_dir, max_chars=max_chars)


def parse_table_ref(table_ref: str) -> tuple[str, int]:
    """`<doc_name>|<số dòng>` -> (doc_name, line)."""
    doc_name, _, position = table_ref.rpartition("|")
    if not doc_name or not position.isdigit():
        raise ValueError(f"table_ref không hợp lệ: {table_ref!r} (cần dạng '<doc_name>|<số dòng>')")
    return doc_name, int(position)


def csv_filename(table_ref: str) -> str:
    """`AAA_..._2015_consolidated|350` -> `AAA_..._2015_consolidated_table_350.csv` (350 = số dòng).

    Dùng chung cho cả bước execute (Kaggle) và bước đóng gói `data/` trong submission.zip,
    để `csv_path` trong `evidence` luôn trỏ đúng file đã dùng khi tính đáp án.
    """
    doc_name, line = parse_table_ref(table_ref)
    return f"{doc_name}_table_{line}.csv"


def build_index_text(table: TableAsset, company_name: str = "", *, max_chars: int = 1200) -> str:
    """Text đưa vào BM25: metadata + ngữ cảnh trước bảng + header + nhãn dòng (bỏ ô số).

    Ô số bị loại khỏi nhãn dòng vì BM25 trên chuỗi số OCR gần như chỉ thêm nhiễu; câu hỏi
    hầu như luôn khớp theo tên chỉ tiêu (chữ), không theo giá trị.
    """
    company_part = f"Công ty: {company_name} (mã {table.ticker})" if company_name else f"Mã: {table.ticker}"
    scope_part = f", phạm vi {table.scope}" if table.scope else ""
    meta = f"{company_part}, năm {table.year}{scope_part}, báo cáo {table.doc_name}."
    columns = " | ".join(h for h in table.header if h)
    labels: list[str] = []
    for row in table.rows[:40]:
        text_cells = [cell for cell in row[:3] if cell and not _looks_numeric(cell)]
        if text_cells:
            labels.append(" / ".join(text_cells))
    body = f"Ngữ cảnh: {table.context_before} Cột: {columns}. Dòng: {' | '.join(labels)}"
    return f"{meta} {body}"[:max_chars]


def _looks_numeric(cell: str) -> bool:
    stripped = cell.strip().strip("()%")
    if not stripped:
        return False
    return all(ch.isdigit() or ch in ".,-  " for ch in stripped)


def to_index_entry(table: TableAsset, company_name: str = "") -> TableIndexEntry:
    return TableIndexEntry(
        table_ref=table.table_ref,
        doc_name=table.doc_name,
        ticker=table.ticker,
        year=table.year,
        scope=table.scope,
        line=table.line,
        order=table.order,
        page=table.page,
        n_rows=table.n_rows,
        n_cols=table.n_cols,
        index_text=build_index_text(table, company_name),
    )
