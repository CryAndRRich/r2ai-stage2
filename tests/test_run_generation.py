"""Test vòng lặp generation phía Kaggle — chạy không cần GPU (dry-run / stub query)."""

from __future__ import annotations

import json

import pytest

from r2ai.generation.run_generation import (
    DEFAULT_STUB_QUERY,
    existing_ids,
    load_retrieval_results,
    materialize_csvs,
    run,
)
from r2ai.schemas import RetrievalCandidate, RetrievalResult

CSV_TEXT = "Chỉ tiêu,Số cuối năm\nTiền,1.234\nĐầu tư,2.000\nTổng,3.234\n"


@pytest.fixture()
def retrieval_file(tmp_path):
    path = tmp_path / "retrieval_results.jsonl"
    lines = []
    for qid in (1, 2):
        result = RetrievalResult(
            id=qid,
            question=f"Câu hỏi {qid}?",
            tickers=["NKG"],
            years=[2022],
            scope="consolidated",
            candidates=[
                RetrievalCandidate(
                    table_ref=f"NKG_financial_statements_2022_consolidated|{qid}",
                    doc_name="NKG_financial_statements_2022_consolidated",
                    ticker="NKG",
                    year=2022,
                    scope="consolidated",
                    page=8,
                    score=10.0,
                    rank=0,
                    csv_text=CSV_TEXT,
                )
            ],
        )
        lines.append(result.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_path_helpers_accept_plain_strings(retrieval_file, tmp_path):
    """Notebook Kaggle giữ đường dẫn dạng `str` — gọi trực tiếp không được `AttributeError`.

    Lỗi thật gặp trên Kaggle: cell smoke-test gọi `load_retrieval_results(RETRIEVAL_PATH)` với chuỗi
    -> `'str' object has no attribute 'exists'` (chỉ `run()` mới tự bọc `Path`).
    """
    assert len(load_retrieval_results(str(retrieval_file))) == 2
    assert existing_ids(str(tmp_path / "chua-ton-tai.jsonl")) == set()


def test_shard_splits_questions_without_overlap_or_loss(retrieval_file, tmp_path):
    """Chạy song song 2 GPU: mỗi shard làm một nửa câu, hợp lại phải đủ và không trùng."""
    seen: list[int] = []
    for shard in (0, 1):
        out = tmp_path / f"pred{shard}.jsonl"
        stats = run(
            retrieval_path=retrieval_file,
            out_path=out,
            work_dir=tmp_path / f"exec{shard}",
            dry_run=True,
            shard_index=shard,
            shard_count=2,
        )
        assert stats["total"] == 1  # fixture có id 1 và 2 -> mỗi shard đúng 1 câu
        seen.extend(row["id"] for row in _rows(out))
    assert sorted(seen) == [1, 2]  # đủ, không trùng


def test_shard_index_out_of_range_is_rejected(retrieval_file, tmp_path):
    with pytest.raises(ValueError, match="shard_index"):
        run(
            retrieval_path=retrieval_file,
            out_path=tmp_path / "pred.jsonl",
            work_dir=tmp_path / "exec",
            dry_run=True,
            shard_index=2,
            shard_count=2,
        )


def test_dry_run_actually_executes_sandbox(retrieval_file, tmp_path):
    """Bug 5: trước đây dry-run để query rỗng nên sandbox bị short-circuit, không test được gì."""
    out = tmp_path / "predictions.jsonl"
    stats = run(retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "exec", dry_run=True)
    assert stats["attempted"] == 2
    assert stats["exec_ok"] == 2, stats
    rows = _rows(out)
    assert all(row["pandas_query"] == DEFAULT_STUB_QUERY for row in rows)
    assert all(row["exec_error"] is None for row in rows)
    # Stub đếm số ô số ở cột cuối -> 3 dòng dữ liệu đều là số.
    assert {row["answer"] for row in rows} == {3.0}
    assert all(row["evidence"][0]["csv_path"].startswith("data/") for row in rows)


def test_dry_run_reports_sandbox_error_instead_of_empty_query(retrieval_file, tmp_path):
    out = tmp_path / "predictions.jsonl"
    run(
        retrieval_path=retrieval_file,
        out_path=out,
        work_dir=tmp_path / "exec",
        dry_run=True,
        stub_query="import os\nresult = 1",
    )
    rows = _rows(out)
    assert all(not row["exec_ok"] for row in rows)
    assert all("import" in row["exec_error"] for row in rows)


def test_stub_query_without_dry_run_skips_model(retrieval_file, tmp_path):
    """`--stub-query` cũng phải chạy được mà không nạp LLM (nếu nạp sẽ lỗi vì không có transformers)."""
    out = tmp_path / "predictions.jsonl"
    stats = run(
        retrieval_path=retrieval_file,
        out_path=out,
        work_dir=tmp_path / "exec",
        stub_query="result = float(len(df1))",
    )
    assert stats["exec_ok"] == 2
    assert {row["answer"] for row in _rows(out)} == {3.0}


def test_resume_skips_finished_ids(retrieval_file, tmp_path):
    out = tmp_path / "predictions.jsonl"
    run(retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "exec", dry_run=True, limit=1)
    assert existing_ids(out) == {1}
    stats = run(retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "exec", dry_run=True)
    assert stats["skipped_done"] == 1
    assert stats["attempted"] == 1
    assert {row["id"] for row in _rows(out)} == {1, 2}


class _CrashingLLM:
    """Giả lập OOM/lỗi generate thật gặp trên Kaggle — `.complete()` luôn raise."""

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("CUDA out of memory (simulated)")


def test_llm_crash_does_not_kill_whole_run(retrieval_file, tmp_path):
    """1 câu OOM/lỗi generate không được làm chết cả 1.012 câu — ghi nhận lỗi rồi qua câu sau."""
    out = tmp_path / "predictions.jsonl"
    stats = run(llm=_CrashingLLM(), retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "exec")
    assert stats["attempted"] == 2
    assert stats["exec_ok"] == 0
    assert stats["exec_failed"] == 2
    rows = _rows(out)
    assert len(rows) == 2  # cả 2 câu đều được ghi, không mất câu nào vì crash
    assert all(not row["exec_ok"] for row in rows)
    assert all("LLM generate lỗi" in (row["exec_error"] or "") for row in rows)
    assert all("RuntimeError" in (row["exec_error"] or "") for row in rows)
    assert all(row["pandas_query"] == "" for row in rows)


@pytest.fixture()
def retrieval_file_6_tables(tmp_path):
    """1 câu, 6 candidate — để giảm `max_tables` thật sự làm prompt ngắn lại."""
    path = tmp_path / "retrieval6.jsonl"
    result = RetrievalResult(
        id=1,
        question="Câu hỏi nhiều bảng?",
        tickers=["NKG"],
        years=[2022],
        candidates=[
            RetrievalCandidate(
                table_ref=f"NKG_financial_statements_2022_consolidated|{100 + n}",
                doc_name="NKG_financial_statements_2022_consolidated",
                ticker="NKG",
                year=2022,
                scope="consolidated",
                line=100 + n,
                rank=n,
                score=10.0 - n,
                csv_text=CSV_TEXT * 8,  # đủ dài để mỗi bảng đóng góp đáng kể vào prompt
            )
            for n in range(6)
        ],
    )
    path.write_text(result.model_dump_json() + "\n", encoding="utf-8")
    return path


class _OomUntilShortPrompt:
    """Giả lập OOM thật: chỉ generate được khi prompt đã ngắn lại (ít bảng hơn)."""

    def __init__(self, limit_chars: int) -> None:
        self.limit_chars = limit_chars
        self.calls: list[int] = []

    def complete(self, system: str, user: str) -> str:
        size = len(system) + len(user)
        self.calls.append(size)
        if size > self.limit_chars:
            raise RuntimeError(
                f"CUDA out of memory. Tried to allocate 4.79 GiB (prompt {size} ký tự)"
            )
        return "result = float(len(df1))"


def test_oom_retries_with_shorter_prompt_and_recovers(retrieval_file_6_tables, tmp_path):
    """OOM phụ thuộc độ dài prompt -> phải thử lại với ít bảng hơn thay vì bỏ trắng câu."""
    probe = _OomUntilShortPrompt(limit_chars=0)  # OOM mọi lần, chỉ để đo kích thước từng bậc
    run(llm=probe, retrieval_path=retrieval_file_6_tables, out_path=tmp_path / "p0.jsonl",
        work_dir=tmp_path / "e0", limit=1, resume=False)
    assert len(probe.calls) == 4, probe.calls  # thang retry 6 -> 3 -> 2 -> 1
    assert probe.calls == sorted(probe.calls, reverse=True), probe.calls  # prompt ngắn dần thật

    # Ngưỡng nằm giữa bậc 1 và bậc 2 -> lần thử đầu OOM, lần thử thứ 2 (3 bảng) phải thành công.
    llm = _OomUntilShortPrompt(limit_chars=(probe.calls[0] + probe.calls[1]) // 2)
    out = tmp_path / "p1.jsonl"
    stats = run(llm=llm, retrieval_path=retrieval_file_6_tables, out_path=out,
                work_dir=tmp_path / "e1", limit=1, resume=False)
    assert stats["generate_oom"] == 1, stats  # OOM đúng 1 lần (ở prompt đầy đủ)
    assert stats["generate_retried"] == 1, stats  # rồi dựng lại prompt ngắn hơn 1 lần
    assert stats["exec_ok"] == 1, stats  # và CỨU được câu đó
    rows = _rows(out)
    assert len(rows) == 1 and rows[0]["exec_ok"] and rows[0]["answer"] is not None


def test_non_oom_error_is_not_retried(retrieval_file, tmp_path):
    """Lỗi không phải OOM thì retry vô nghĩa — chỉ gọi model 1 lần rồi ghi lỗi."""
    class _Broken:
        def __init__(self): self.n = 0
        def complete(self, system, user):
            self.n += 1
            raise ValueError("lỗi tokenizer, không phải OOM")

    llm = _Broken()
    out = tmp_path / "p.jsonl"
    stats = run(llm=llm, retrieval_path=retrieval_file, out_path=out, work_dir=tmp_path / "e",
                limit=1, resume=False)
    assert llm.n == 1 and stats["generate_oom"] == 0 and stats["generate_retried"] == 0
    assert stats["exec_failed"] == 1


def test_materialize_csvs_writes_expected_filenames(tmp_path):
    result = RetrievalResult(
        id=1,
        question="q",
        candidates=[
            RetrievalCandidate(
                table_ref="DOC|7", doc_name="DOC", ticker="X", csv_text=CSV_TEXT
            )
        ],
    )
    paths = materialize_csvs(result, {"df1": "DOC|7"}, tmp_path)
    assert paths["df1"].name == "DOC_table_7.csv"
    assert paths["df1"].read_text(encoding="utf-8") == CSV_TEXT
