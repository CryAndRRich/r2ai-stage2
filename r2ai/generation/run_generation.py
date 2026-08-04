"""CLI (chạy trên Kaggle T4): retrieval_results.jsonl -> predictions.jsonl

    python -m r2ai.generation.run_generation --retrieval retrieval_results.jsonl \
        --out predictions.jsonl --work-dir /kaggle/working/exec --limit 20

Đặc điểm bắt buộc (ARCHITECTURE.md mục 2):
- Ghi `predictions.jsonl` **append + flush sau mỗi câu** -> mất session Kaggle không mất kết quả.
- `--resume` bỏ qua các id đã có trong file predictions -> chạy tiếp được sau khi bị ngắt.
- Thực thi ngay tại chỗ để biết `exec_ok`, nhưng ở local vẫn re-execute lại lần nữa lúc đóng gói.

Model được nạp trễ (import transformers ngay trong hàm) để CLI này còn `--dry-run` được ở local
không GPU. `--dry-run` **không** để query rỗng: nó chạy `DEFAULT_STUB_QUERY` (hoặc `--stub-query`
do người dùng đưa) qua đúng đường thực thi thật — nếu để rỗng thì `run_pandas_code` short-circuit
ở "pandas_query rỗng" và smoke test sẽ bỏ sót đúng phần rủi ro nhất (sandbox + load CSV).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from r2ai.config import load_config
from r2ai.constants import PREDICTIONS_PATH, RETRIEVAL_RESULTS_PATH
from r2ai.execution.numeric import to_float
from r2ai.execution.sandbox import ExecutionResult, run_pandas_code
from r2ai.extraction.table_store import csv_filename
from r2ai.prompting.build_prompt import build_prompt, extract_code, finalize_code, used_variables
from r2ai.schemas import EvidenceItem, Prediction, RetrievalResult

logger = logging.getLogger(__name__)

# Query giả cho `--dry-run`: đọc thật cột cuối của bảng đầu tiên và parse số kiểu VN, nên nó đi
# qua đúng đường mà query thật sẽ đi (AST pre-check -> load CSV dtype=str -> sandbox -> to_float).
DEFAULT_STUB_QUERY = """\
values = []
for cell in df1.iloc[:, -1]:
    text = str(cell).strip().strip("()").replace(".", "").replace(",", ".")
    if text and all(ch.isdigit() or ch == "." for ch in text):
        values.append(float(text))
result = float(len(values))
"""


def _retry_ladder(n_tables: int) -> list[int]:
    """Số bảng cho từng lần thử: đủ ngân sách -> giảm dần -> tối thiểu 1 bảng."""
    ladder = [n_tables, max(3, n_tables // 2), 2, 1]
    out: list[int] = []
    for n in ladder:
        n = max(1, min(n, n_tables))
        if n not in out:
            out.append(n)
    return out


def _is_out_of_memory(exc: BaseException) -> bool:
    """Nhận diện OOM GPU không cần import torch (chạy được cả ở local không có CUDA)."""
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    return "out of memory" in str(exc).lower()


def _free_gpu_memory() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best-effort, không để việc dọn cache làm chết cả run
        pass


def load_retrieval_results(path: Path | str) -> list[RetrievalResult]:
    """Đọc `retrieval_results.jsonl`. Nhận cả `str` — notebook Kaggle giữ đường dẫn dạng chuỗi."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path} — chạy bước retrieval ở local trước.")
    results: list[RetrievalResult] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(RetrievalResult.model_validate_json(line))
    return results


def existing_ids(path: Path | str) -> set[int]:
    path = Path(path)
    if not path.exists():
        return set()
    ids: set[int] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(int(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # dòng ghi dở do bị ngắt giữa chừng -> sẽ được sinh lại
    return ids


def materialize_csvs(result: RetrievalResult, variables: dict[str, str], work_dir: Path) -> dict[str, Path]:
    """Ghi CSV của các bảng trong prompt ra đĩa để sandbox load được. Trả về {variable: path}."""
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    by_ref = {c.table_ref: c for c in result.candidates}
    paths: dict[str, Path] = {}
    for variable, table_ref in variables.items():
        candidate = by_ref.get(table_ref)
        if candidate is None:
            continue
        path = data_dir / csv_filename(table_ref)
        if not path.exists():
            path.write_text(candidate.csv_text, encoding="utf-8")
        paths[variable] = path
    return paths


class LocalLLM:
    """Wrapper transformers (4-bit) — nạp model 1 lần rồi sinh tuần tự từng câu."""

    def __init__(self, model_name: str, *, load_in_4bit: bool, max_new_tokens: int, temperature: float) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # requirements-kaggle.txt chỉ ghim version tối thiểu (>=), không ghim version cứng, nên
        # bản `transformers` thực tế cài được trên Kaggle thay đổi theo thời gian/theo image.
        # Tên kwarg chọn dtype đã đổi 2 lần giữa các bản: `torch_dtype` (cũ) rồi `dtype` (mới hơn,
        # `torch_dtype` bị deprecate). Không đoán cứng 1 tên — thử `dtype` trước (API hiện hành),
        # nếu bản cài được là bản cũ chưa hỗ trợ thì rơi về `torch_dtype`.
        # `device_map={"": 0}` (ép GPU 0 duy nhất), KHÔNG dùng `"auto"`: model 7B 4-bit chỉ ~4-5GB,
        # dư sức nằm trên 1 T4 (16GB). `"auto"` trên session có ≥2 GPU (T4 x2) sẽ CHIA layer model
        # ra nhiều GPU — activation/attention buffer lúc prefill KHÔNG được chia đều theo dung
        # lượng còn trống của từng GPU, nên 1 GPU có thể OOM dù tổng dung lượng cả 2 GPU vẫn dư
        # (đã gặp thật: "GPU 1 ... 10.97 GiB memory in use" trong khi GPU khác còn trống nhiều).
        kwargs: dict = {"device_map": {"": 0}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        # `attn_implementation="sdpa"` — KHÔNG để mặc định. Attention kiểu "eager" materialize hẳn
        # ma trận (heads × seq × seq): với prompt ~6.000 token của pipeline này là 28 × 6000² × 2B
        # ≈ 2GB, upcast fp32 lúc softmax thì ~4-5GB cho MỘT allocation — khớp đúng con số OOM thật
        # đã gặp ("Tried to allocate 4.79 GiB" trong khi model 4-bit chỉ ~4,5GB). SDPA dùng kernel
        # memory-efficient, không materialize ma trận đó; T4 (sm75) không chạy được flash-attn-2
        # nhưng backend memory-efficient của SDPA thì hỗ trợ sm75.
        self.attn_implementation = "default"
        for attn in ("sdpa", None):
            attn_kwargs = {"attn_implementation": attn} if attn else {}
            try:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, dtype=torch.float16, **attn_kwargs, **kwargs
                    )
                except TypeError:  # bản transformers cũ dùng tên `torch_dtype`
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype=torch.float16, **attn_kwargs, **kwargs
                    )
                self.attn_implementation = attn or "default"
                break
            except (ValueError, ImportError) as exc:  # bản/model không nhận sdpa -> để mặc định
                logger.warning("attn_implementation=%s không dùng được (%s) — thử mặc định", attn, exc)
        self.model.eval()
        logger.info("model nạp xong | attention = %s", self.attn_implementation)

        # Qwen ship generation_config có top_p/top_k; ở chế độ greedy chúng vô nghĩa và transformers
        # in cảnh báo "not valid and may be ignored" mỗi câu. Dọn cho log sạch.
        if self.temperature <= 0.0:
            for field in ("top_p", "top_k", "temperature"):
                if hasattr(self.model.generation_config, field):
                    setattr(self.model.generation_config, field, None)
            self.model.generation_config.do_sample = False

        # Chỉ tính logits cho token CUỐI, không cho toàn bộ prompt. Đây là nguyên nhân thật của những
        # lần OOM xin **4,5-4,7 GiB cho một allocation** trong log pilot: lm_head trên toàn prompt là
        # `seq × vocab(152.064) × 4 byte` (fp32 lúc softmax) — với ~8.200 token là đúng 4,67 GiB.
        # transformers mới tự truyền tham số này trong `generate`, bản cũ thì không; tên tham số đã
        # đổi (`num_logits_to_keep` -> `logits_to_keep`) nên dò 1 lần bằng chính chữ ký hàm forward.
        self._logits_kwargs: dict = {}
        try:
            import inspect

            forward_params = inspect.signature(self.model.forward).parameters
            for name in ("logits_to_keep", "num_logits_to_keep"):
                if name in forward_params:
                    self._logits_kwargs = {name: 1}
                    break
        except (TypeError, ValueError):  # chữ ký bị wrap/không đọc được -> để mặc định
            pass
        logger.info("logits_to_keep = %s", self._logits_kwargs or "(không hỗ trợ, để mặc định)")

    def complete(self, system: str, user: str) -> str:
        import gc

        import torch  # type: ignore[import-not-found]

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        greedy = self.temperature <= 0.0
        output = None
        # Dọn TRƯỚC khi generate, không chỉ sau: OOM thật xảy ra ngay lúc xin thêm ~1GB trong khi
        # process đã giữ 13,6-14,4GB — phần lớn là block của câu trước còn nằm trong allocator.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            with torch.inference_mode():
                try:
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=not greedy,
                        temperature=None if greedy else self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        **self._logits_kwargs,
                    )
                except TypeError as exc:
                    # Bản transformers đã tự truyền tham số này -> "got multiple values". Bỏ và nhớ.
                    if not self._logits_kwargs:
                        raise
                    logger.warning("bỏ %s (%s)", self._logits_kwargs, exc)
                    self._logits_kwargs = {}
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=not greedy,
                        temperature=None if greedy else self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        finally:
            del inputs, output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run(
    *,
    retrieval_path: Path | None = None,
    out_path: Path | None = None,
    work_dir: Path | None = None,
    config_path: Path | None = None,
    model_name: str | None = None,
    limit: int | None = None,
    ids: set[int] | None = None,
    resume: bool = True,
    dry_run: bool = False,
    stub_query: str | None = None,
    llm: "LocalLLM | None" = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict:
    config = load_config(config_path)
    gen_cfg = config["generation"]
    retrieval_cfg = config["retrieval"]
    timeout_s = float(config["execution"]["timeout_s"])
    startup_timeout_s = float(config["execution"]["startup_timeout_s"])

    retrieval_file = Path(retrieval_path) if retrieval_path else RETRIEVAL_RESULTS_PATH
    predictions_file = Path(out_path) if out_path else PREDICTIONS_PATH
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    exec_dir = Path(work_dir) if work_dir else predictions_file.parent / "exec"

    results = load_retrieval_results(retrieval_file)
    if ids:
        results = [r for r in results if r.id in ids]
    if shard_count:
        # Chia câu hỏi cho nhiều process chạy song song (1 process / 1 GPU — xem notebook mục 3b).
        # Chia theo `id % shard_count` chứ không cắt đoạn liên tiếp: (1) ổn định khi `--resume`,
        # (2) mỗi shard nhận đều các dạng câu nên tốc độ 2 shard không lệch nhau nhiều.
        if not 0 <= (shard_index or 0) < shard_count:
            raise ValueError(f"shard_index phải trong [0, {shard_count}), nhận {shard_index}")
        results = [r for r in results if r.id % shard_count == (shard_index or 0)]
    done = existing_ids(predictions_file) if resume else set()
    pending = [r for r in results if r.id not in done]
    if limit is not None:
        pending = pending[:limit]

    if dry_run and not stub_query:
        stub_query = DEFAULT_STUB_QUERY

    # `llm` có thể được truyền sẵn (đã load 1 lần) để gọi `run()` nhiều lần (pilot rồi full) trong
    # cùng 1 process Jupyter mà không phải nạp lại model 7B từ đầu mỗi lần — xem notebooks/kaggle_generate.ipynb.
    if llm is None and not dry_run and not stub_query:
        llm = LocalLLM(
            model_name or gen_cfg["model"],
            load_in_4bit=bool(gen_cfg["load_in_4bit"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
        )

    stats = {
        "total": len(results),
        "skipped_done": len(done),
        "attempted": 0,
        "exec_ok": 0,
        "exec_failed": 0,
        "generate_oom": 0,  # số lần generate hết bộ nhớ GPU (có thể nhiều lần cho cùng 1 câu)
        "generate_retried": 0,  # số lần phải dựng lại prompt ngắn hơn rồi generate lại
    }

    with predictions_file.open("a", encoding="utf-8") as out:
        for i, result in enumerate(pending, start=1):
            n_tables = int(retrieval_cfg["candidates_in_prompt"])
            prompt = build_prompt(
                result,
                max_tables=n_tables,
                max_csv_chars=int(retrieval_cfg["max_csv_chars"]),
                max_prompt_chars=int(gen_cfg["max_prompt_chars"]),
            )
            generation_error: str | None = None
            if llm is not None:
                # OOM phụ thuộc ĐỘ DÀI PROMPT (bộ nhớ attention tăng theo seq), nên khi OOM thì thử
                # lại với ít bảng hơn thay vì bỏ trắng câu: mất 1-2 bảng ít liên quan vẫn tốt hơn
                # mất cả câu. Chỉ retry với lỗi hết bộ nhớ — lỗi khác (bug code, tokenizer...) thì
                # retry chỉ tốn thời gian.
                for attempt, tables in enumerate(_retry_ladder(n_tables)):
                    if attempt:
                        prompt = build_prompt(
                            result,
                            max_tables=tables,
                            max_csv_chars=int(retrieval_cfg["max_csv_chars"]),
                            max_prompt_chars=int(gen_cfg["max_prompt_chars"]),
                        )
                        stats["generate_retried"] += 1
                        logger.warning(
                            "[%d/%d] id=%s thử lại với %d bảng (prompt %d ký tự)",
                            i, len(pending), result.id, tables, len(prompt.system) + len(prompt.user),
                        )
                    try:
                        completion = llm.complete(prompt.system, prompt.user)
                        code = finalize_code(extract_code(completion))
                        generation_error = None
                        break
                    except Exception as exc:  # noqa: BLE001 - OOM/lỗi generate khác không được làm
                        # chết cả 1.012 câu; ghi nhận lỗi cho câu này rồi qua câu sau.
                        completion = ""
                        code = ""
                        generation_error = f"LLM generate lỗi: {type(exc).__name__}: {exc}"
                        oom = _is_out_of_memory(exc)
                        stats["generate_oom"] += 1 if oom else 0
                        logger.warning(
                            "[%d/%d] id=%s generate lỗi (%d bảng, oom=%s): %s",
                            i, len(pending), result.id, tables, oom, generation_error,
                        )
                        _free_gpu_memory()
                        if not oom:
                            break
            else:
                # Không có model: chạy query giả để **đường thực thi thật vẫn được kiểm tra**
                # (AST pre-check + load CSV + sandbox + ép kiểu kết quả). Nếu để code rỗng thì
                # `run_pandas_code` short-circuit ngay ở "pandas_query rỗng" và bước smoke test
                # này chẳng kiểm tra được gì ngoài việc dựng prompt.
                completion = ""
                code = stub_query
            variables = {
                var: ref for var, ref in prompt.variables.items() if var in used_variables(code, prompt.variables)
            } or prompt.variables
            csv_paths = materialize_csvs(result, variables, exec_dir)

            if generation_error is not None:
                execution = ExecutionResult(ok=False, error=generation_error)
            else:
                execution = run_pandas_code(
                    code, dict(csv_paths), timeout_s=timeout_s, startup_timeout_s=startup_timeout_s
                )
            answer = to_float(execution.value) if execution.ok else None
            prediction = Prediction(
                id=result.id,
                pandas_query=code,
                answer=answer,
                evidence=[
                    EvidenceItem(variable=var, csv_path=f"data/{csv_filename(ref)}")
                    for var, ref in variables.items()
                ],
                used_table_refs=list(variables.values()),
                exec_ok=bool(execution.ok and answer is not None),
                exec_error=execution.error
                or (None if answer is not None else f"result không phải số: {execution.value!r}"),
                raw_completion=completion[:4000] or None,
            )
            out.write(prediction.model_dump_json() + "\n")
            out.flush()  # chống mất dữ liệu khi Kaggle ngắt session giữa chừng

            stats["attempted"] += 1
            stats["exec_ok"] += 1 if prediction.exec_ok else 0
            stats["exec_failed"] += 0 if prediction.exec_ok else 1
            if i % 10 == 0 or i == len(pending):
                logger.info(
                    "[%d/%d] exec_ok=%d exec_failed=%d", i, len(pending), stats["exec_ok"], stats["exec_failed"]
                )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sinh + thực thi pandas_query -> predictions.jsonl")
    parser.add_argument("--retrieval", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None, help="nơi ghi CSV tạm để thực thi")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None, help="override tên model trong config")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--no-resume", action="store_true", help="sinh lại cả những id đã có")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="không nạp LLM; chạy query giả qua sandbox để smoke test toàn luồng prompt->CSV->exec",
    )
    parser.add_argument(
        "--stub-query",
        default=None,
        help="pandas_query cố định dùng thay output LLM (mặc định khi --dry-run: DEFAULT_STUB_QUERY)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help='chia việc cho nhiều process song song, dạng "i/n" (vd "0/2" và "1/2" cho 2 GPU)',
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    ids = {int(p) for p in args.ids.replace(" ", "").split(",") if p} if args.ids else None
    shard_index = shard_count = None
    if args.shard:
        raw_index, _, raw_count = args.shard.partition("/")
        if not raw_count.isdigit():
            parser.error(f'--shard phải có dạng "i/n" (vd "0/2"), nhận {args.shard!r}')
        shard_index, shard_count = int(raw_index), int(raw_count)
    stats = run(
        retrieval_path=args.retrieval,
        out_path=args.out,
        work_dir=args.work_dir,
        config_path=args.config,
        model_name=args.model,
        limit=args.limit,
        ids=ids,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        stub_query=args.stub_query,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
