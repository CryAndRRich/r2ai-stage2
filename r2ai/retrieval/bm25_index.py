"""Build/load index BM25 (`bm25s`) trên `tables_index.jsonl`, cache tại `data/interim/bm25_index/`.

Cache invalidate theo: fingerprint file tables_index.jsonl (size+mtime), version tokenizer,
version format index, và tên tokenizer thực tế đã dùng.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from r2ai.constants import BM25_INDEX_DIR, BM25_TOKENIZER_VERSION, INDEX_FORMAT_VERSION, TABLES_INDEX_PATH
from r2ai.retrieval.tokenize import tokenize, using_underthesea
from r2ai.schemas import TableIndexEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Hit:
    entry: TableIndexEntry
    score: float


def load_index_entries(path: Path | None = None) -> list[TableIndexEntry]:
    index_path = Path(path) if path else TABLES_INDEX_PATH
    if not index_path.exists():
        raise FileNotFoundError(
            f"Chưa có {index_path} — chạy `python -m r2ai.extraction.build_table_index` trước."
        )
    entries: list[TableIndexEntry] = []
    with index_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(TableIndexEntry.model_validate_json(line))
    return entries


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


class BM25Index:
    def __init__(self, model, entries: list[TableIndexEntry]) -> None:
        self._model = model
        self.entries = entries

    @property
    def size(self) -> int:
        return len(self.entries)

    def search(self, query: str, *, top_k: int) -> list[Hit]:
        if not self.entries or self._model is None or top_k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        k = min(top_k, len(self.entries))
        results, scores = self._model.retrieve([tokens], k=k, show_progress=False)
        return [
            Hit(entry=self.entries[int(idx)], score=float(score))
            for idx, score in zip(results[0], scores[0], strict=True)
        ]


def _manifest(index_path: Path) -> dict:
    return {
        "index_format_version": INDEX_FORMAT_VERSION,
        "tokenizer_version": BM25_TOKENIZER_VERSION,
        "tokenizer": "underthesea" if using_underthesea() else "regex_fallback",
        "fingerprint": _fingerprint(index_path),
    }


def load_or_build(
    *,
    index_path: Path | None = None,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> BM25Index:
    import bm25s  # import trễ: chỉ bước retrieval cần, tránh làm nặng các CLI khác

    tables_index_path = Path(index_path) if index_path else TABLES_INDEX_PATH
    out_dir = Path(cache_dir) if cache_dir else BM25_INDEX_DIR
    manifest_path = out_dir / "manifest.json"
    expected = _manifest(tables_index_path)
    entries = load_index_entries(tables_index_path)

    if not rebuild and manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = None
        if stored == expected:
            logger.info("BM25 cache HIT (%d bảng)", len(entries))
            return BM25Index(bm25s.BM25.load(str(out_dir), load_corpus=False), entries)

    logger.info("BM25 cache MISS -> tokenize + index %d bảng (chậm, chỉ chạy lại khi index đổi)", len(entries))
    if not entries:
        return BM25Index(None, [])
    # Log tiến độ: tokenize ~135K bảng bằng underthesea mất khoảng 7 phút và `bm25s` chỉ hiện
    # progress bar ở bước index hoá SAU đó — không log thì lần chạy đầu trông y như bị treo.
    corpus_tokens: list[list[str]] = []
    step = max(1, len(entries) // 20)
    for i, entry in enumerate(entries, start=1):
        corpus_tokens.append(tokenize(entry.index_text))
        if i % step == 0 or i == len(entries):
            logger.info("  tokenize %d/%d bảng (%.0f%%)", i, len(entries), 100 * i / len(entries))
    model = bm25s.BM25()
    model.index(corpus_tokens, show_progress=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    manifest_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return BM25Index(model, entries)
