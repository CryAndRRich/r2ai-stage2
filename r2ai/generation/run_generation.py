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
from r2ai.execution.sandbox import run_pandas_code
from r2ai.extraction.table_store import csv_filename
from r2ai.prompting.build_prompt import build_prompt, extract_code, used_variables
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


def load_retrieval_results(path: Path) -> list[RetrievalResult]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path} — chạy bước retrieval ở local trước.")
    results: list[RetrievalResult] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(RetrievalResult.model_validate_json(line))
    return results


def existing_ids(path: Path) -> set[int]:
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
        # `torch_dtype`, KHÔNG phải `dtype`: transformers 4.46.3 (bản ghim trong requirements-kaggle)
        # dùng tên `torch_dtype`; tên `dtype` chỉ có ở bản mới hơn nhiều và sẽ bị nuốt thành config
        # lạ -> model load fp32. Với 4-bit thì bnb che mất triệu chứng, nhưng tắt 4-bit là OOM T4.
        kwargs: dict = {"torch_dtype": torch.float16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def complete(self, system: str, user: str) -> str:
        import torch  # type: ignore[import-not-found]

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        greedy = self.temperature <= 0.0
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else self.temperature,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


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
    done = existing_ids(predictions_file) if resume else set()
    pending = [r for r in results if r.id not in done]
    if limit is not None:
        pending = pending[:limit]

    if dry_run and not stub_query:
        stub_query = DEFAULT_STUB_QUERY

    llm = None
    if not dry_run and not stub_query:
        llm = LocalLLM(
            model_name or gen_cfg["model"],
            load_in_4bit=bool(gen_cfg["load_in_4bit"]),
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
        )

    stats = {"total": len(results), "skipped_done": len(done), "attempted": 0, "exec_ok": 0, "exec_failed": 0}

    with predictions_file.open("a", encoding="utf-8") as out:
        for i, result in enumerate(pending, start=1):
            prompt = build_prompt(
                result,
                max_tables=int(retrieval_cfg["candidates_in_prompt"]),
                max_csv_chars=int(retrieval_cfg["max_csv_chars"]),
                max_prompt_chars=int(gen_cfg["max_prompt_chars"]),
            )
            if llm is not None:
                completion = llm.complete(prompt.system, prompt.user)
                code = extract_code(completion)
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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    ids = {int(p) for p in args.ids.replace(" ", "").split(",") if p} if args.ids else None
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
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
