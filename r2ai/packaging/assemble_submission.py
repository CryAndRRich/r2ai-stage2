"""CLI: join retrieval_results + predictions -> submission.json + data/*.csv (re-execute ở local).

    python -m r2ai.packaging.assemble_submission --predictions data/interim/predictions.jsonl

Quy tắc (COMPETITION.md mục 3 + ARCHITECTURE.md mục 2):
- Phải đủ id 1..N của questions.jsonl; câu thiếu -> vẫn ghi item với answer fallback (bài nộp
  thiếu câu sẽ không được đánh giá).
- `answer` bắt buộc là float.
- `relevant_tables` lấy top-N candidate của retrieval (F2 beta=2 nghiêng về recall), luôn gồm
  các bảng thực sự dùng trong `pandas_query`.
- `relevant_docs` suy ra từ `relevant_tables`.
- **Re-execute lại toàn bộ query ở local** trên đúng file CSV sẽ nộp — không tin kết quả Kaggle
  (lệch version pandas/numpy giữa 2 môi trường).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from r2ai.config import load_config
from r2ai.constants import (
    PREDICTIONS_PATH,
    QUESTIONS_PATH,
    RETRIEVAL_RESULTS_PATH,
    SUBMISSION_BUILD_DIR,
)
from r2ai.execution.numeric import to_float
from r2ai.execution.sandbox import run_pandas_code
from r2ai.extraction.table_store import csv_filename, parse_table_ref
from r2ai.generation.run_generation import load_retrieval_results
from r2ai.packaging.zip_submission import validate_build
from r2ai.retrieval.run_retrieval import load_questions
from r2ai.schemas import EvidenceItem, Prediction, RetrievalResult, SubmissionItem

logger = logging.getLogger(__name__)


def load_predictions(path: Path) -> dict[int, Prediction]:
    """Đọc predictions.jsonl; dòng sau ghi đè dòng trước cùng id (lần chạy lại là bản mới hơn)."""
    predictions: dict[int, Prediction] = {}
    if not path.exists():
        return predictions
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                prediction = Prediction.model_validate_json(line)
            except Exception as exc:  # dòng ghi dở khi Kaggle bị ngắt
                logger.warning("bỏ dòng predictions không đọc được (dòng %d): %s", line_no, exc)
                continue
            predictions[prediction.id] = prediction
    return predictions


def relevant_tables_for(
    result: RetrievalResult | None, prediction: Prediction | None, *, n_tables: int
) -> list[str]:
    """Bảng thực sự dùng trong query đứng trước, sau đó bù thêm candidate top BM25."""
    refs: list[str] = []
    if prediction is not None:
        refs.extend(prediction.used_table_refs)
    if result is not None:
        refs.extend(c.table_ref for c in result.candidates)
    out: list[str] = []
    for ref in refs:
        if ref not in out:
            out.append(ref)
        if len(out) >= n_tables:
            break
    return out


def write_csv_assets(
    result: RetrievalResult | None, table_refs: list[str], data_dir: Path
) -> dict[str, Path]:
    """Ghi CSV vào `<build>/data/` cho **đúng những bảng được `evidence` tham chiếu**.

    KHÔNG ghi CSV cho mọi bảng trong `relevant_tables`: field đó chỉ cần id `<doc>|<dòng>`, BTC
    chấm retrieval theo id chứ không đọc file. Ghi cả 5 bảng/câu × 1.012 câu sẽ nhồi vào zip hàng
    nghìn CSV mà không câu nào dùng tới.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    by_ref = {c.table_ref: c for c in (result.candidates if result else [])}
    written: dict[str, Path] = {}
    for ref in table_refs:
        candidate = by_ref.get(ref)
        csv_text = candidate.csv_text if candidate else ""
        if not csv_text:
            # Fallback: đọc lại từ cache extraction ở local (retrieval có thể đã cắt bớt CSV).
            from r2ai.extraction.table_store import load_table_csv

            doc_name, line = parse_table_ref(ref)
            csv_text = load_table_csv(doc_name, line)
        if not csv_text:
            logger.warning("không có nội dung CSV cho %s — bỏ khỏi evidence", ref)
            continue
        path = data_dir / csv_filename(ref)
        path.write_text(csv_text, encoding="utf-8")
        written[ref] = path
    return written


def _ref_for_csv_name(csv_name: str, prediction: Prediction | None, table_refs: list[str]) -> str | None:
    """Tìm `table_ref` tương ứng với tên file CSV mà `evidence` trỏ tới (None nếu không khớp bảng nào)."""
    known = dict.fromkeys([*(prediction.used_table_refs if prediction else []), *table_refs])
    return next((ref for ref in known if csv_filename(ref) == csv_name), None)


def evidence_refs_for(prediction: Prediction | None, table_refs: list[str]) -> list[str]:
    """Bảng cần có file CSV: những bảng `evidence` của prediction trỏ tới (đã bỏ trùng).

    Nếu prediction không có evidence hợp lệ (câu thiếu prediction, hoặc query lỗi), lấy tạm bảng
    top-1 để `evidence` không rỗng.
    """
    if prediction is None:
        return table_refs[:1]
    refs: list[str] = []
    for item in prediction.evidence:
        ref = _ref_for_csv_name(Path(item.csv_path).name, prediction, table_refs)
        if ref is not None and ref not in refs:
            refs.append(ref)
    return refs or table_refs[:1]


def assemble(
    *,
    questions_path: Path | None = None,
    retrieval_path: Path | None = None,
    predictions_path: Path | None = None,
    build_dir: Path | None = None,
    config_path: Path | None = None,
    reexecute: bool = True,
) -> dict:
    config = load_config(config_path)
    n_tables = int(config["retrieval"]["submission_tables"])
    timeout_s = float(config["execution"]["timeout_s"])
    startup_timeout_s = float(config["execution"]["startup_timeout_s"])
    fallback_answer = float(config["answer"]["fallback_answer"])

    questions = load_questions(questions_path)
    results = {r.id: r for r in load_retrieval_results(Path(retrieval_path or RETRIEVAL_RESULTS_PATH))}
    predictions = load_predictions(Path(predictions_path or PREDICTIONS_PATH))

    out_dir = Path(build_dir) if build_dir else SUBMISSION_BUILD_DIR
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob("*.csv"):
        stale.unlink()  # tránh mang theo CSV của lần build trước (zip sẽ phình + lệch evidence)

    stats = {
        "questions": len(questions),
        "with_prediction": 0,
        "reexecuted_ok": 0,
        "reexecuted_failed": 0,
        "reexecute_skipped": 0,
        "answer_changed_vs_kaggle": 0,
        "fallback_answer_used": 0,
        "empty_query": 0,
        "duplicate_evidence_variables": 0,
        "kaggle_answer_not_verified": 0,
    }
    # Danh sách item chưa thật sự được "double-check" ở local — không nhét được vào submission.json
    # (schema BTC cố định) nên ghi ra file riêng để soi trước khi nộp.
    reexec_report: list[dict] = []
    items: list[SubmissionItem] = []

    for question in questions:
        qid = int(question["id"])
        result = results.get(qid)
        prediction = predictions.get(qid)
        if prediction is not None:
            stats["with_prediction"] += 1

        # `relevant_tables` = id các bảng nộp (không cần file); chỉ bảng nào `evidence` dùng mới ghi CSV.
        table_refs = relevant_tables_for(result, prediction, n_tables=n_tables)

        # Lọc duplicate tên biến TRƯỚC khi ghi CSV — nếu ghi trước rồi mới loại evidence trùng thì
        # CSV của entry bị loại nằm lại trong `data/` như file mồ côi, không ai tham chiếu tới.
        plan: list[tuple[str, str]] = []  # (variable, table_ref)
        taken: set[str] = set()
        for item in prediction.evidence if prediction else []:
            ref = _ref_for_csv_name(Path(item.csv_path).name, prediction, table_refs)
            if ref is None:
                continue
            if item.variable in taken:
                # BTC cấm trùng tên biến trong cùng 1 câu hỏi — chặn ngay tại đây thay vì đợi
                # `validate_build()` bắt, để `assemble()` gọi trực tiếp cũng không sinh file lỗi.
                stats["duplicate_evidence_variables"] += 1
                logger.warning("id=%d trùng biến evidence `%s` — bỏ bản sau", qid, item.variable)
                continue
            taken.add(item.variable)
            plan.append((item.variable, ref))

        needed = list(dict.fromkeys(ref for _, ref in plan)) or table_refs[:1]
        written = write_csv_assets(result, needed, data_dir)

        query = prediction.pandas_query if prediction else ""
        evidence: list[EvidenceItem] = []
        csv_paths: dict[str, Path] = {}
        for variable, ref in plan:
            path = written.get(ref)
            if path is None:  # không dựng được CSV cho bảng này -> bỏ khỏi evidence
                continue
            evidence.append(EvidenceItem(variable=variable, csv_path=f"data/{path.name}"))
            csv_paths[variable] = path

        answer = prediction.answer if prediction and prediction.answer is not None else None
        reexec_status = "skipped"
        if reexecute and query and csv_paths:
            execution = run_pandas_code(
                query, dict(csv_paths), timeout_s=timeout_s, startup_timeout_s=startup_timeout_s
            )
            local_answer = to_float(execution.value) if execution.ok else None
            if local_answer is None:
                reexec_status = "failed"
                stats["reexecuted_failed"] += 1
                logger.warning("id=%d re-execute lỗi ở local: %s", qid, execution.error)
                reexec_report.append(
                    {"id": qid, "status": "failed", "kaggle_answer": answer, "error": execution.error}
                )
            else:
                reexec_status = "ok"
                stats["reexecuted_ok"] += 1
                if answer is not None and abs(local_answer - answer) > 1e-6:
                    stats["answer_changed_vs_kaggle"] += 1
                    logger.warning(
                        "id=%d lệch kết quả Kaggle(%r) vs local(%r) — lấy local", qid, answer, local_answer
                    )
                    reexec_report.append(
                        {
                            "id": qid,
                            "status": "mismatch_fixed",
                            "kaggle_answer": answer,
                            "local_answer": local_answer,
                        }
                    )
                answer = local_answer
        elif reexecute:
            stats["reexecute_skipped"] += 1
            if answer is not None:
                # Giữ nguyên answer từ Kaggle nhưng ghi rõ là **chưa được xác minh lại ở local**.
                reexec_report.append(
                    {
                        "id": qid,
                        "status": "skipped",
                        "kaggle_answer": answer,
                        "reason": "không có pandas_query" if not query else "không có CSV cho evidence",
                    }
                )

        if answer is not None and reexec_status != "ok":
            stats["kaggle_answer_not_verified"] += 1

        if not query:
            stats["empty_query"] += 1
        if answer is None:
            answer = fallback_answer
            stats["fallback_answer_used"] += 1

        if not evidence:
            # `evidence` không được rỗng nếu còn bảng nào ghi được: gán df1 cho bảng đầu tiên.
            first = next(iter(written.items()), None)
            if first is not None:
                evidence = [EvidenceItem(variable="df1", csv_path=f"data/{first[1].name}")]

        items.append(
            SubmissionItem(
                id=qid,
                question=question["question"],
                answer=float(answer),
                # relevant_docs/relevant_tables lấy theo **id retrieval** (không phụ thuộc việc có
                # ghi CSV hay không) — chấm retrieval chỉ so id, còn CSV chỉ phục vụ pandas_query.
                relevant_docs=list(dict.fromkeys(parse_table_ref(ref)[0] for ref in table_refs)),
                relevant_tables=list(table_refs),
                evidence=evidence,
                pandas_query=query,
            )
        )

    submission_path = out_dir / "submission.json"
    submission_path.write_text(
        json.dumps([item.model_dump() for item in items], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    stats["items"] = len(items)
    stats["csv_files"] = len(list(data_dir.glob("*.csv")))
    (out_dir / "assemble_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "reexec_report.json").write_text(
        json.dumps(reexec_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if stats["kaggle_answer_not_verified"]:
        logger.warning(
            "%d câu giữ nguyên answer từ Kaggle mà CHƯA re-execute thành công ở local — xem reexec_report.json",
            stats["kaggle_answer_not_verified"],
        )

    # Self-check: `assemble()` không được sinh ra thư mục build sai định dạng, kể cả khi gọi trực
    # tiếp (không qua CLI zip_submission). Chỉ log, không raise — để còn soi được output lỗi.
    errors = validate_build(out_dir, questions_path=Path(questions_path) if questions_path else QUESTIONS_PATH)
    stats["validation_errors"] = len(errors)
    if errors:
        logger.error("build có %d lỗi validate (5 lỗi đầu): %s", len(errors), "; ".join(errors[:5]))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đóng gói submission.json + data/*.csv")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--retrieval", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-reexecute", action="store_true", help="bỏ bước re-execute ở local (không khuyến khích)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    stats = assemble(
        questions_path=args.questions,
        retrieval_path=args.retrieval,
        predictions_path=args.predictions,
        build_dir=args.build_dir,
        config_path=args.config,
        reexecute=not args.no_reexecute,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
