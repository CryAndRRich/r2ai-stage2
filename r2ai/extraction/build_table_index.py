"""CLI: extract toàn bộ corpus -> `data/interim/tables_index.jsonl` (+ cache theo report).

Chạy:
    python -m r2ai.extraction.build_table_index                 # dùng cache, chỉ extract file đổi
    python -m r2ai.extraction.build_table_index --rebuild       # bỏ cache, extract lại từ đầu
    python -m r2ai.extraction.build_table_index --limit-docs 20 # smoke test nhanh

Bước này là bước tốn CPU nhất của pipeline local (~1.973 file, ~362MB text).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from r2ai.config import load_config
from r2ai.constants import (
    CODE_STOCK_PATH,
    STATEMENTS_DIR,
    TABLES_CACHE_DIR,
    TABLES_INDEX_PATH,
)
from r2ai.extraction.context import DocumentContext
from r2ai.extraction.doc_scanner import scan_documents
from r2ai.extraction.html_tables import check_tag_balance, locate_tables, parse_table, split_header
from r2ai.extraction.table_store import fingerprint, read_cache, to_index_entry, write_cache
from r2ai.retrieval.company_meta import load_companies
from r2ai.schemas import DocumentRef, TableAsset

logger = logging.getLogger(__name__)


def _looks_numeric(cell: str) -> bool:
    stripped = cell.strip().strip("()%")
    if not stripped:
        return False
    return any(ch.isdigit() for ch in stripped) and all(
        ch.isdigit() or ch in ".,-  " for ch in stripped
    )


def is_eligible(grid: list[list[str]], *, min_cells: int, require_numeric: bool) -> bool:
    """Lọc bảng vô dụng khỏi index: bảng quá nhỏ, hoặc không có ô số nào (danh sách nhân sự...)."""
    non_empty = [cell for row in grid for cell in row if cell]
    if len(non_empty) < min_cells:
        return False
    if require_numeric and not any(_looks_numeric(cell) for cell in non_empty):
        return False
    return True


def extract_document(doc: DocumentRef, *, line_base: int = 1) -> tuple[list[list[list[str]]], list[dict]]:
    """Parse mọi bảng của 1 report, kèm **số dòng bắt đầu** (khoá `<vị trí>` của `table_ref`).

    `line_base=0` chuyển sang đếm dòng từ 0 (xem `configs/baseline.yaml`). Bảng không parse được
    vẫn giữ chỗ bằng grid rỗng để không mất vị trí trong dãy.
    """
    text = Path(doc.path).read_text(encoding="utf-8", errors="replace")
    context = DocumentContext(text)
    grids: list[list[list[str]]] = []
    metas: list[dict] = []
    for raw in locate_tables(text, source=doc.doc_name):
        line = raw.line - (1 - line_base)
        grid = parse_table(raw.html, source=f"{doc.doc_name}|{line}")
        grids.append(grid)
        metas.append(
            {
                "line": line,
                "order": raw.index,
                "page": context.page_at(raw.start),
                "context_before": context.text_before(raw.start),
                "closed": raw.closed,
            }
        )
    return grids, metas


def _fp(doc: DocumentRef) -> dict[str, int] | None:
    try:
        return fingerprint(Path(doc.path)).as_dict()
    except OSError:
        return None


def _entries_from_payload(
    doc: DocumentRef, payload: dict, company_name: str, *, min_cells: int, require_numeric: bool
) -> list[dict]:
    entries: list[dict] = []
    for item in payload["tables"]:
        grid = item["grid"]
        if not is_eligible(grid, min_cells=min_cells, require_numeric=require_numeric):
            continue
        header, rows = split_header(grid)
        table = TableAsset(
            doc_name=doc.doc_name,
            ticker=doc.ticker,
            year=doc.year,
            scope=doc.scope,
            line=item["line"],
            order=item.get("order", 0),
            page=item.get("page"),
            context_before=item.get("context_before", ""),
            header=header,
            rows=rows,
        )
        entries.append(to_index_entry(table, company_name).model_dump())
    return entries


def build(
    *,
    statements_dir: Path = STATEMENTS_DIR,
    cache_dir: Path = TABLES_CACHE_DIR,
    index_path: Path = TABLES_INDEX_PATH,
    company_meta_path: Path = CODE_STOCK_PATH,
    config_path: Path | None = None,
    rebuild: bool = False,
    limit_docs: int | None = None,
) -> dict:
    config = load_config(config_path)
    retrieval_cfg = config["retrieval"]
    min_cells = int(retrieval_cfg["min_table_cells"])
    require_numeric = bool(retrieval_cfg["require_numeric"])
    line_base = int(config["extraction"]["table_ref_line_base"])

    companies = load_companies(company_meta_path)
    docs = scan_documents(statements_dir)
    if limit_docs is not None:
        docs = docs[:limit_docs]

    index_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "documents": len(docs),
        "table_ref_line_base": line_base,
        "documents_extracted": 0,
        "documents_from_cache": 0,
        "tables_located": 0,
        "tables_parsed": 0,
        "tables_indexed": 0,
        "tables_unparsed": 0,
        "tables_unclosed": 0,  # bảng thiếu `</table>` -> giữ chỗ bằng grid rỗng
        "docs_with_unbalanced_tags": 0,
        "docs_without_scope": 0,
        "docs_without_year": 0,
    }

    tmp_path = index_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for i, doc in enumerate(docs, start=1):
            company_name = companies[doc.ticker].name if doc.ticker in companies else ""
            if doc.scope is None:
                stats["docs_without_scope"] += 1
            if doc.year is None:
                stats["docs_without_year"] += 1

            payload: dict | None = None
            if not rebuild:
                cached = read_cache(doc.doc_name, cache_dir=cache_dir)
                if cached is not None and cached.get("fingerprint") == _fp(doc):
                    payload = cached
                    stats["documents_from_cache"] += 1

            if payload is None:
                text = Path(doc.path).read_text(encoding="utf-8", errors="replace")
                n_open, n_close = check_tag_balance(text)
                if n_open != n_close:
                    stats["docs_with_unbalanced_tags"] += 1
                del text
                grids, metas = extract_document(doc, line_base=line_base)
                write_cache(doc, grids, metas, cache_dir=cache_dir)
                stats["documents_extracted"] += 1
                payload = {
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
                    ]
                }

            stats["tables_located"] += len(payload["tables"])
            stats["tables_parsed"] += sum(1 for t in payload["tables"] if t["grid"])
            stats["tables_unparsed"] += sum(1 for t in payload["tables"] if not t["grid"])
            stats["tables_unclosed"] += sum(1 for t in payload["tables"] if not t.get("closed", True))
            entries = _entries_from_payload(
                doc, payload, company_name, min_cells=min_cells, require_numeric=require_numeric
            )

            for entry in entries:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stats["tables_indexed"] += len(entries)

            if i % 50 == 0 or i == len(docs):
                logger.info(
                    "[%d/%d] indexed=%d located=%d cache_hit=%d",
                    i,
                    len(docs),
                    stats["tables_indexed"],
                    stats["tables_located"],
                    stats["documents_from_cache"],
                )
    tmp_path.replace(index_path)

    stats_path = index_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract bảng HTML từ corpus BCTC -> tables_index.jsonl")
    parser.add_argument("--statements-dir", type=Path, default=STATEMENTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=TABLES_CACHE_DIR)
    parser.add_argument("--index-path", type=Path, default=TABLES_INDEX_PATH)
    parser.add_argument("--company-meta", type=Path, default=CODE_STOCK_PATH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true", help="bỏ qua cache, extract lại toàn bộ")
    parser.add_argument("--limit-docs", type=int, default=None, help="chỉ xử lý N report đầu (smoke test)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    stats = build(
        statements_dir=args.statements_dir,
        cache_dir=args.cache_dir,
        index_path=args.index_path,
        company_meta_path=args.company_meta,
        config_path=args.config,
        rebuild=args.rebuild,
        limit_docs=args.limit_docs,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
