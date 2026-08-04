"""Đường dẫn & hằng số dùng chung.

Mọi đường dẫn đều suy ra từ vị trí file này (`<repo>/code/r2ai/constants.py`) nên pipeline
chạy được từ bất kỳ CWD nào. Có thể override bằng biến môi trường khi chạy trên Kaggle
(nơi không có sẵn `data/` gốc 362MB).
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/code/r2ai/constants.py -> parents[2] == <repo>
CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


DATA_DIR = _env_path("R2AI_DATA_DIR", PROJECT_ROOT / "data")
STATEMENTS_DIR = DATA_DIR / "financial_statements"
QUESTIONS_PATH = DATA_DIR / "questions" / "questions.jsonl"
CODE_STOCK_PATH = DATA_DIR / "code_stock.csv"

INTERIM_DIR = _env_path("R2AI_INTERIM_DIR", DATA_DIR / "interim")
TABLES_CACHE_DIR = INTERIM_DIR / "tables_cache"
TABLES_INDEX_PATH = INTERIM_DIR / "tables_index.jsonl"
BM25_INDEX_DIR = INTERIM_DIR / "bm25_index"
RETRIEVAL_RESULTS_PATH = INTERIM_DIR / "retrieval_results.jsonl"
PREDICTIONS_PATH = INTERIM_DIR / "predictions.jsonl"

OUT_DIR = _env_path("R2AI_OUT_DIR", PROJECT_ROOT / "out")
SUBMISSION_BUILD_DIR = OUT_DIR / "submission"
SUBMISSION_ZIP_PATH = OUT_DIR / "submission.zip"

CONFIG_DIR = CODE_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "baseline.yaml"
TEMPLATES_DIR = CODE_ROOT / "r2ai" / "prompting" / "templates"

# Suffix mà BTC thêm vào tên file OCR: <doc_name>_extracted.txt
EXTRACTED_SUFFIX = "_extracted.txt"

# Marker phân trang trong file OCR: "===== PAGE 12 ====="
PAGE_MARKER_PATTERN = r"=====\s*PAGE\s+(\d+)\s*====="

# Khoảng năm hợp lệ của corpus (DATA_DESCRIPTION.md mục 3.1) — dùng để lọc năm parse từ câu hỏi.
MIN_YEAR = 2015
MAX_YEAR = 2025

SCOPE_CONSOLIDATED = "consolidated"
SCOPE_SEPARATE = "separate"
SCOPE_AGGREGATED = "aggregated"
SCOPES = (SCOPE_CONSOLIDATED, SCOPE_SEPARATE, SCOPE_AGGREGATED)

# Tăng khi đổi logic extraction / tokenizer — dùng để invalidate cache.
# v2: locate_tables() quét theo từng tag `<table` mở (bảng thiếu `</table>` giữ chỗ thay vì
#     nuốt bảng kế tiếp) — cache v1 có thể lệch số thứ tự nên phải extract lại.
# v3: `<vị trí>` trong table_ref đổi từ thứ tự xuất hiện sang **số dòng bắt đầu bảng** (BTC xác nhận
#     ở discussion) — cache v2 không có field `line` nên phải extract lại.
EXTRACTION_VERSION = 3
BM25_TOKENIZER_VERSION = 1
INDEX_FORMAT_VERSION = 1

# Giới hạn colspan/rowspan: OCR đôi khi sinh ra giá trị rác (colspan="9999").
MAX_SPAN = 40
# Bỏ qua bảng quá lớn (gần chắc chắn là lỗi parse) để không làm nổ bộ nhớ.
MAX_TABLE_CELLS = 20_000

# Ngưỡng khớp đáp án — BTC trả lời ở discussion: "Ngưỡng sai số cho phép là không quá 0,02% so với
# đáp án", tức sai số **TƯƠNG ĐỐI** 2e-4, KHÔNG phải sai số tuyệt đối 1e-2 như bản v1 (suy từ công
# thức `math.isclose(..., abs_tol=1e-2)` của repo ViFinQA tham khảo). Khác biệt quan trọng theo 2
# chiều: với số tiền lớn thì 0,02% rộng hơn 0,01 rất nhiều, còn với đáp án nhỏ (hệ số/lần, tỉ lệ)
# thì 0,02% chặt hơn — nên KHÔNG được làm tròn đáp án về 2 chữ số thập phân nữa (xem system prompt).
ANSWER_REL_TOL = 2e-4
ANSWER_ABS_TOL = 0.0
