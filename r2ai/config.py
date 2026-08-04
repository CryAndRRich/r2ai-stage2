"""Đọc `configs/baseline.yaml` và merge lên default (deep-merge, không ghi đè cả nhánh)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from r2ai.constants import ANSWER_ABS_TOL, ANSWER_REL_TOL, DEFAULT_CONFIG_PATH

DEFAULTS: dict[str, Any] = {
    "extraction": {
        # `<vị trí>` trong `table_ref` = số dòng bắt đầu bảng. BTC nói "số line bắt đầu bảng" nhưng
        # không nói đếm từ 0 hay 1; để 1 (cách đọc thông thường của "dòng"). Nếu leaderboard vẫn cho
        # Tables F2 = 0 thì đổi thành 0 rồi chạy lại extraction + retrieval, không phải sửa code.
        "table_ref_line_base": 1,
    },
    "retrieval": {
        "top_k": 20,  # số candidate ghi vào retrieval_results.jsonl
        "fetch_multiplier": 6,  # over-fetch trước khi lọc metadata
        "fallback_quota": 2,  # giữ lại vài hit không khớp metadata (chống parse ticker sai)
        "candidates_in_prompt": 4,  # số bảng nhúng vào prompt LLM
        "submission_tables": 5,  # số table_ref nộp ở relevant_tables (F2 ưu tiên recall)
        "max_csv_chars": 6000,  # cắt CSV nhúng trong retrieval_results/prompt
        "year_expand": True,  # "cuối năm N" cũng có thể nằm ở report N+1 (số đầu năm)
        "min_table_cells": 4,  # bỏ bảng quá nhỏ khỏi index
        "require_numeric": True,  # bảng phải có ít nhất 1 ô số
    },
    "generation": {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "fallback_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "load_in_4bit": True,
        "max_new_tokens": 320,  # code cần sinh ~10-20 dòng; xem ghi chú trong configs/baseline.yaml
        "temperature": 0.0,
        "max_prompt_chars": 24000,
    },
    "execution": {
        "timeout_s": 20,  # chỉ tính phần code chạy, không tính khởi động process con
        "startup_timeout_s": 120,  # hạn riêng cho spawn + import pandas trong process con
    },
    "answer": {
        "abs_tol": ANSWER_ABS_TOL,
        "rel_tol": ANSWER_REL_TOL,
        "fallback_answer": 0.0,  # dùng khi query lỗi — submission bắt buộc có answer float
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return copy.deepcopy(DEFAULTS)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config phải là mapping ở top level: {config_path}")
    return _deep_merge(DEFAULTS, loaded)
