"""CLI: questions.jsonl + tables_index.jsonl -> data/interim/retrieval_results.jsonl

Chạy:
    python -m r2ai.retrieval.run_retrieval                      # toàn bộ 1.012 câu
    python -m r2ai.retrieval.run_retrieval --limit 20           # pilot
    python -m r2ai.retrieval.run_retrieval --ids 1,5,99

Mỗi dòng output nhúng sẵn CSV của các bảng candidate để bước generation trên Kaggle không cần
mount corpus gốc 362MB (xem ARCHITECTURE.md mục 2).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from r2ai.config import load_config
from r2ai.constants import (
    BM25_INDEX_DIR,
    CODE_STOCK_PATH,
    QUESTIONS_PATH,
    RETRIEVAL_RESULTS_PATH,
    TABLES_CACHE_DIR,
    TABLES_INDEX_PATH,
)
from r2ai.extraction.table_store import load_table, parse_table_ref, table_to_csv
from r2ai.retrieval.bm25_index import load_or_build
from r2ai.retrieval.company_meta import load_companies
from r2ai.retrieval.metadata_filter import parse_question_meta, select_candidates
from r2ai.schemas import RetrievalCandidate, RetrievalResult

logger = logging.getLogger(__name__)


def load_questions(path: Path | None = None) -> list[dict]:
    questions_path = Path(path) if path else QUESTIONS_PATH
    if not questions_path.exists():
        raise FileNotFoundError(f"Không tìm thấy questions.jsonl: {questions_path}")
    questions: list[dict] = []
    with questions_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    questions.sort(key=lambda q: int(q["id"]))
    return questions


def run(
    *,
    questions_path: Path | None = None,
    index_path: Path | None = None,
    bm25_cache_dir: Path | None = None,
    tables_cache_dir: Path | None = None,
    company_meta_path: Path | None = None,
    output_path: Path | None = None,
    config_path: Path | None = None,
    limit: int | None = None,
    ids: set[int] | None = None,
    rebuild_index: bool = False,
) -> dict:
    config = load_config(config_path)["retrieval"]
    top_k = int(config["top_k"])
    fetch_n = top_k * int(config["fetch_multiplier"])
    fallback_quota = int(config["fallback_quota"])
    max_csv_chars = int(config["max_csv_chars"])
    year_expand = bool(config["year_expand"])

    companies = load_companies(company_meta_path or CODE_STOCK_PATH)
    index = load_or_build(
        index_path=index_path or TABLES_INDEX_PATH,
        cache_dir=bm25_cache_dir or BM25_INDEX_DIR,
        rebuild=rebuild_index,
    )
    questions = load_questions(questions_path)
    if ids:
        questions = [q for q in questions if int(q["id"]) in ids]
    if limit is not None:
        questions = questions[:limit]

    out_path = Path(output_path) if output_path else RETRIEVAL_RESULTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = tables_cache_dir or TABLES_CACHE_DIR

    stats = {
        "questions": len(questions),
        "index_size": index.size,
        "empty_candidates": 0,
        "questions_with_ticker": 0,
        "questions_with_year": 0,
        "questions_with_scope": 0,
        "top1_score_zero": 0,
    }

    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for i, question in enumerate(questions, start=1):
            qid = int(question["id"])
            text = question["question"]
            meta = parse_question_meta(text, companies)
            if meta.tickers:
                stats["questions_with_ticker"] += 1
            if meta.years:
                stats["questions_with_year"] += 1
            if meta.scope:
                stats["questions_with_scope"] += 1

            hits = index.search(text, top_k=fetch_n)
            selected = select_candidates(
                hits, meta, top_k=top_k, fallback_quota=fallback_quota, year_expand=year_expand
            )
            if not selected:
                stats["empty_candidates"] += 1
                logger.warning("câu hỏi id=%d không có candidate nào (recall chắc chắn 0)", qid)
            elif selected[0].score <= 0.0:
                stats["top1_score_zero"] += 1

            candidates: list[RetrievalCandidate] = []
            for rank, hit in enumerate(selected):
                entry = hit.entry
                doc_name, line = parse_table_ref(entry.table_ref)
                table = load_table(doc_name, line, cache_dir=cache_dir)
                if table is None:
                    logger.warning("thiếu cache cho %s — bỏ candidate", entry.table_ref)
                    continue
                candidates.append(
                    RetrievalCandidate(
                        table_ref=entry.table_ref,
                        doc_name=entry.doc_name,
                        ticker=entry.ticker,
                        year=entry.year,
                        scope=entry.scope,
                        line=entry.line,
                        page=entry.page,
                        score=hit.score,
                        rank=rank,
                        context_before=table.context_before,
                        csv_text=table_to_csv(table, max_chars=max_csv_chars),
                    )
                )
            result = RetrievalResult(
                id=qid,
                question=text,
                tickers=sorted(meta.tickers),
                years=sorted(meta.years),
                scope=meta.scope,
                candidates=candidates,
            )
            out.write(result.model_dump_json() + "\n")
            if i % 100 == 0 or i == len(questions):
                logger.info("[%d/%d] retrieval xong", i, len(questions))
    tmp_path.replace(out_path)

    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part) for part in raw.replace(" ", "").split(",") if part}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BM25 retrieval + lọc metadata -> retrieval_results.jsonl")
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--bm25-cache-dir", type=Path, default=None)
    parser.add_argument("--tables-cache-dir", type=Path, default=None)
    parser.add_argument("--company-meta", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="danh sách id, phân tách bằng dấu phẩy")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    stats = run(
        questions_path=args.questions,
        index_path=args.index_path,
        bm25_cache_dir=args.bm25_cache_dir,
        tables_cache_dir=args.tables_cache_dir,
        company_meta_path=args.company_meta,
        output_path=args.out,
        config_path=args.config,
        limit=args.limit,
        ids=_parse_ids(args.ids),
        rebuild_index=args.rebuild_index,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
