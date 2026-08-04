# r2ai-stage2

Pipeline **Table Retrieval + Text-to-Pandas** cho cuộc thi R2AI2026 (BCTC doanh nghiệp niêm yết VN).
Thiết kế đầy đủ: [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) · luật thi: [`../docs/COMPETITION.md`](../docs/COMPETITION.md) · dữ liệu: [`../docs/DATA_DESCRIPTION.md`](../docs/DATA_DESCRIPTION.md).

## Kiến trúc 2 giai đoạn

```
[Local, CPU]                                     [Kaggle Notebook, T4 GPU]
extraction → BM25 retrieval → retrieval_results.jsonl ──upload──► generation (LLM) + execute
                                                                          │
submission.zip ◄── package + re-execute ◄── download ── predictions.jsonl ┘
```

## Cài đặt

```bash
pip install -r requirements.txt   # 1 file duy nhất, dùng cho cả local (CPU) và Kaggle Notebook
```

Mọi CLI chạy từ thư mục `code/`: `python -m r2ai.<module>`. Không cần `pip install -e .`.

## Chạy pipeline

```bash
# 1. Extract bảng HTML từ toàn bộ corpus -> data/interim/tables_index.jsonl (có cache theo report)
python -m r2ai.extraction.build_table_index            # thêm --limit-docs 20 để smoke test

# 2. BM25 retrieval + lọc metadata -> data/interim/retrieval_results.jsonl
python -m r2ai.retrieval.run_retrieval                 # thêm --limit 20 để pilot

# 3. Sanity check retrieval (không cần gold)
python -m r2ai.sanity.retrieval_probe --show 10

# 4. Sinh + thực thi pandas_query (chạy trên Kaggle — xem notebooks/kaggle_generate.ipynb)
python -m r2ai.generation.run_generation --dry-run --limit 5   # local: chỉ test prompt + sandbox
# Chạy song song nhiều GPU: 1 process/GPU, mỗi process 1 shard (chia theo id % n), rồi gộp file lại.
# CUDA_VISIBLE_DEVICES=0 python -m r2ai.generation.run_generation --shard 0/2 --out pred0.jsonl &
# CUDA_VISIBLE_DEVICES=1 python -m r2ai.generation.run_generation --shard 1/2 --out pred1.jsonl &

# 5. Đóng gói (re-execute lại toàn bộ query ở local) + validate + zip
python -m r2ai.packaging.assemble_submission
python -m r2ai.packaging.zip_submission
```

Test: `pytest tests/` (chạy từ `code/`).

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| `r2ai/extraction/` | `html_tables.py` (lxml, colspan/rowspan) · `doc_scanner.py` · `context.py` (page marker + text trước bảng) · `table_store.py` (cache theo report) · `build_table_index.py` (CLI) |
| `r2ai/retrieval/` | `tokenize.py` (underthesea + fallback) · `bm25_index.py` (bm25s + cache) · `metadata_filter.py` (ticker/năm/scope) · `company_meta.py` · `run_retrieval.py` (CLI) |
| `r2ai/prompting/` | `templates/` (system + user) · `hints.py` (tính sẵn: cột ↔ kỳ, đơn vị, hệ số đổi) · `build_prompt.py` (nhúng CSV + chú thích, gán `df1..dfN`, hậu xử lý output) |
| `r2ai/execution/` | `sandbox.py` (AST pre-check + builtins whitelist + timeout process) · `numeric.py` (số kiểu VN) |
| `r2ai/generation/` | `run_generation.py` (CLI Kaggle: LLM 4-bit, ghi predictions append+flush, `--resume`) |
| `r2ai/packaging/` | `assemble_submission.py` (join + re-execute) · `zip_submission.py` (validate + zip) |
| `r2ai/sanity/` | `retrieval_probe.py` (kiểm tra retrieval không cần gold) |
| `configs/baseline.yaml` | top_k, số bảng nộp, tên model, timeout, tolerance |

## Giả định cần ghi vào working-notes paper

- **`table_ref` = `<doc_name>|<số dòng bắt đầu bảng>`** (dòng chứa tag `<table` mở, đếm từ 1) — theo BTC trả lời
  ở discussion: *"vị trí bảng ở đây là số line bắt đầu bảng trong file ocr báo cáo tương ứng"*. Bản nộp thử dùng
  **thứ tự bảng** cho Docs F2 = 0,6678 nhưng Tables F2 = 0,0000, xác nhận format `doc|N` đúng còn `N` sai.
  BTC không nói đếm từ 0 hay 1 → mặc định 1, đổi bằng `extraction.table_ref_line_base` trong `configs/baseline.yaml`
  (đổi xong phải chạy lại `build_table_index` + `run_retrieval`).
- **`pandas_query` phải tự chứa**: sandbox chỉ cấp `pd` + builtins whitelist (kèm vài tên lớp exception),
  không inject helper parse số của mình — vì BTC có thể re-execute query trong môi trường của họ, nơi
  helper đó không tồn tại. Thay vì bắt LLM tự chép lại hàm `to_num()` mỗi câu (tốn ~250 token, dễ bị cắt
  cụt khi hết `max_new_tokens`), `finalize_code()` (`build_prompt.py`) **tự ghép** `to_num()` (nối chuỗi
  văn bản, `numeric.TO_NUM_HELPER_SOURCE`) vào trước code LLM sinh ra — `pandas_query` cuối cùng vẫn tự
  chứa 100%, chỉ là LLM không cần tốn token viết lại nó.
- **Gợi ý cột/đơn vị nhúng trong prompt là do pipeline tính bằng luật** (`prompting/hints.py`), không phải model
  suy: mỗi bảng có 2 dòng `<!-- Cột theo kỳ: … -->` và `<!-- Đổi đơn vị: … -->` nói rõ cột nào ứng với kỳ được
  hỏi và phải nhân hệ số nào. Đây là bản sửa Bug 15 (nguyên nhân chính khiến V1 chỉ đạt Answer Accuracy 1,4%).
  Khi không đủ bằng chứng thì chú thích ghi "chưa xác định được" — cố ý không đoán.
- **Sửa `split_header()` phải chạy lại index + retrieval**: hàm này chạy lúc *đọc* cache nên không cần extract
  lại corpus (`EXTRACTION_VERSION` không đổi), nhưng `index_text` (BM25) và `csv_text` (nhúng prompt) đều dẫn
  xuất từ header → phải `build_table_index` (đọc cache, ~1 phút) rồi `run_retrieval` lại.
- **`relevant_tables` chỉ là id, không kèm file**: `data/*.csv` trong zip chỉ chứa bảng mà `evidence` tham chiếu
  (bảng thực sự dùng để tính `answer`), không phải toàn bộ bảng đã retrieve.
- **Model**: Qwen2.5-Coder-7B-Instruct (open-weight, 11/2024, ≤14B — hợp lệ theo luật thi).
