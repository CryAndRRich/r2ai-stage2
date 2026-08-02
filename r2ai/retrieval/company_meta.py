"""Load `data/code_stock.csv` (100 dòng: `Mã CK`, `Tên công ty`) + sinh alias để so khớp tên."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from r2ai.constants import CODE_STOCK_PATH
from r2ai.extraction.doc_scanner import ascii_compact

# Tiền tố pháp lý cần bỏ khi sinh alias ("CTCP Tập đoàn Hòa Phát" -> "tapdoanhoaphat").
_LEGAL_PREFIXES = (
    "congtycophantapdoan",
    "congtycophan",
    "ctcptapdoan",
    "ctcp",
    "congtytnhh",
    "tnhh",
    "nganhangtmcp",
    "nganhang",
    "tongcongty",
    "tapdoan",
)
_MIN_ALIAS_LEN = 6


@dataclass(frozen=True, slots=True)
class CompanyInfo:
    ticker: str
    name: str
    aliases: tuple[str, ...]


def _aliases(name: str) -> tuple[str, ...]:
    compact = ascii_compact(name)
    if not compact:
        return ()
    aliases = [compact]
    for prefix in _LEGAL_PREFIXES:
        if compact.startswith(prefix):
            suffix = compact[len(prefix) :]
            if len(suffix) >= _MIN_ALIAS_LEN:
                aliases.append(suffix)
            break
    return tuple(dict.fromkeys(aliases))


def load_companies(path: Path | None = None) -> dict[str, CompanyInfo]:
    csv_path = Path(path) if path else CODE_STOCK_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy code_stock.csv: {csv_path}")
    companies: dict[str, CompanyInfo] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("Mã CK") or "").strip().upper()
            name = (row.get("Tên công ty") or "").strip()
            if not ticker:
                continue
            companies[ticker] = CompanyInfo(ticker=ticker, name=name, aliases=_aliases(name))
    return companies


@lru_cache(maxsize=4)
def get_companies(path_str: str | None = None) -> dict[str, CompanyInfo]:
    return load_companies(Path(path_str) if path_str else None)
