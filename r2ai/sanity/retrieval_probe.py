"""Sanity check retrieval **không cần gold** (ARCHITECTURE.md mục 5.2–5.3).

    python -m r2ai.sanity.retrieval_probe --retrieval data/interim/retrieval_results.jsonl
    python -m r2ai.sanity.retrieval_probe --show 5      # in ra 5 câu để mắt kiểm tra thủ công

Đo 4 nhóm tín hiệu:
1. **Ticker precision @k** trên các câu có mã CK trong ngoặc (229 câu — nhãn miễn phí): candidate
   top-k có đúng ticker đó không.
2. **Phân bố điểm BM25 top-1**: nhóm điểm gần 0 = retrieval thất bại, cần soi.
3. **Coverage**: câu hỏi có candidate set rỗng (rỗng = recall chắc chắn 0).
4. **Khớp năm / khớp scope** giữa metadata parse từ câu hỏi và candidate top-1.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from r2ai.constants import CODE_STOCK_PATH, RETRIEVAL_RESULTS_PATH
from r2ai.generation.run_generation import load_retrieval_results
from r2ai.retrieval.company_meta import load_companies
from r2ai.retrieval.metadata_filter import parse_question_meta
from r2ai.schemas import RetrievalResult


_PARENS_RE = re.compile(r"\([^)]*\)")


def _strip_parens(question: str) -> str:
    """Bỏ mọi nội dung trong ngoặc — mô phỏng câu hỏi không kèm mã CK sẵn."""
    return _PARENS_RE.sub(" ", question)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "min": ordered[0],
        "p10": at(0.10),
        "p25": at(0.25),
        "median": statistics.median(ordered),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": ordered[-1],
    }


def probe(
    results: list[RetrievalResult], *, companies_path: Path | None = None, top_k_list: tuple[int, ...] = (1, 3, 5, 10)
) -> dict:
    companies = load_companies(companies_path or CODE_STOCK_PATH)
    report: dict = {
        "questions": len(results),
        "empty_candidates": 0,
        "empty_candidate_ids": [],
        "candidates_per_question": {},
        "top1_score": {},
        "top1_score_zero": 0,
        "ticker_hit_rate": {},
        "with_paren_ticker": 0,
        "parser_vs_paren": {"agree": 0, "missing": 0, "superset": 0, "conflict": 0},
        # Ticker parser tự thêm ngoài ticker trong ngoặc — mỗi ticker thừa ăn mất một suất trong
        # ngân sách top_k rất hẹp, nên phải soi được chứ không gộp chung vào "superset".
        "extra_tickers_total": 0,
        "extra_ticker_examples": [],
        "extra_ticker_counts": {},
        # Ticker thừa đó có thật sự kéo bảng của công ty sai vào top-k không (hại thực tế).
        "questions_with_wrong_ticker_in_top5": 0,
        "top1_year_match": 0,
        "top1_scope_match": 0,
        "with_year": 0,
        "with_scope": 0,
    }

    top1_scores: list[float] = []
    n_candidates: list[float] = []
    ticker_hits = {k: 0 for k in top_k_list}
    ticker_total = 0

    for result in results:
        n_candidates.append(len(result.candidates))
        if not result.candidates:
            report["empty_candidates"] += 1
            if len(report["empty_candidate_ids"]) < 20:
                report["empty_candidate_ids"].append(result.id)
            continue
        top1 = result.candidates[0]
        top1_scores.append(top1.score)
        if top1.score <= 0.0:
            report["top1_score_zero"] += 1

        meta = parse_question_meta(result.question, companies)
        if meta.ticker_in_parens:
            report["with_paren_ticker"] += 1
            ticker_total += 1
            for k in top_k_list:
                refs = {c.ticker for c in result.candidates[:k]}
                if refs & meta.ticker_in_parens:
                    ticker_hits[k] += 1
            # Kiểm chứng miễn phí (mục 5.2): bỏ phần trong ngoặc rồi xem parser có tự tìm ra
            # đúng ticker đó qua tên công ty / mã trần hay không. So trực tiếp với `meta.tickers`
            # là vô nghĩa vì `meta.tickers` đã chứa sẵn ticker trong ngoặc.
            blind = parse_question_meta(_strip_parens(result.question), companies)
            if blind.tickers == meta.ticker_in_parens:
                report["parser_vs_paren"]["agree"] += 1
            elif meta.ticker_in_parens <= blind.tickers:
                report["parser_vs_paren"]["superset"] += 1
            elif not blind.tickers:
                report["parser_vs_paren"]["missing"] += 1
            else:
                report["parser_vs_paren"]["conflict"] += 1

            # Không gộp mọi "superset" làm một: liệt kê đúng ticker nào bị thêm vào, và đo xem
            # nó có thật sự đẩy bảng của công ty sai vào top-5 hay không.
            extras = sorted(meta.tickers - meta.ticker_in_parens)
            if extras:
                report["extra_tickers_total"] += len(extras)
                for extra in extras:
                    report["extra_ticker_counts"][extra] = report["extra_ticker_counts"].get(extra, 0) + 1
                if len(report["extra_ticker_examples"]) < 20:
                    report["extra_ticker_examples"].append(
                        {
                            "id": result.id,
                            "in_parens": sorted(meta.ticker_in_parens),
                            "extra": extras,
                            "question": result.question[:120],
                        }
                    )
                if any(c.ticker in set(extras) for c in result.candidates[:5]):
                    report["questions_with_wrong_ticker_in_top5"] += 1

        if meta.years:
            report["with_year"] += 1
            if top1.year in meta.years:
                report["top1_year_match"] += 1
        if meta.scope:
            report["with_scope"] += 1
            if top1.scope == meta.scope:
                report["top1_scope_match"] += 1

    report["candidates_per_question"] = _percentiles(n_candidates)
    report["top1_score"] = _percentiles(top1_scores)
    report["ticker_hit_rate"] = {
        f"top{k}": (ticker_hits[k] / ticker_total if ticker_total else None) for k in top_k_list
    }
    return report


def show_examples(results: list[RetrievalResult], count: int, *, per_question: int = 3) -> None:
    """In vài câu để spot-check thủ công (mục 5.3: ~20-30 câu trải đều các dạng)."""
    step = max(1, len(results) // max(1, count))
    for result in results[::step][:count]:
        print("=" * 100)
        print(f"id={result.id} | tickers={result.tickers} years={result.years} scope={result.scope}")
        print(result.question)
        for candidate in result.candidates[:per_question]:
            print(
                f"  [{candidate.rank}] score={candidate.score:.2f} {candidate.table_ref} "
                f"({candidate.ticker}/{candidate.year}/{candidate.scope}, trang {candidate.page})"
            )
            print(f"      ngữ cảnh: {candidate.context_before[:160]}")
            print(f"      csv: {candidate.csv_text[:160]!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanity check retrieval không cần gold")
    parser.add_argument("--retrieval", type=Path, default=RETRIEVAL_RESULTS_PATH)
    parser.add_argument("--company-meta", type=Path, default=CODE_STOCK_PATH)
    parser.add_argument("--show", type=int, default=0, help="in N câu để spot-check thủ công")
    parser.add_argument("--out", type=Path, default=None, help="ghi report ra file JSON")
    args = parser.parse_args(argv)

    results = load_retrieval_results(args.retrieval)
    report = probe(results, companies_path=args.company_meta)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.show:
        show_examples(results, args.show)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
