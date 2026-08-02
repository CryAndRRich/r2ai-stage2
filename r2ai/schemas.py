"""Schema dữ liệu trao đổi giữa các bước pipeline (pydantic v2).

Luồng: DocumentRef -> TableAsset -> TableIndexEntry (tables_index.jsonl)
       -> RetrievalCandidate/RetrievalResult (retrieval_results.jsonl)
       -> Prediction (predictions.jsonl) -> SubmissionItem (submission.json)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class DocumentRef(_Base):
    """Một báo cáo tài chính (1 file `.txt`)."""

    doc_name: str  # id báo cáo theo COMPETITION.md mục 3 (tên file, bỏ .txt / _extracted)
    ticker: str
    year: int | None = None
    scope: str | None = None  # consolidated | separate | aggregated | None
    path: str  # đường dẫn tuyệt đối tới file OCR


class TableAsset(_Base):
    """Một bảng đã parse từ HTML nhúng, kèm ngữ cảnh xung quanh."""

    doc_name: str
    ticker: str
    year: int | None = None
    scope: str | None = None
    line: int  # số dòng bắt đầu bảng trong file OCR -> chính là `<vị trí>` của table_ref
    order: int = 0  # thứ tự xuất hiện 1-indexed (thông tin phụ, để debug/đối chiếu)
    page: int | None = None
    context_before: str = ""
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)  # KHÔNG gồm header

    @property
    def table_ref(self) -> str:
        return f"{self.doc_name}|{self.line}"

    @property
    def n_rows(self) -> int:
        return len(self.rows) + (1 if self.header else 0)

    @property
    def n_cols(self) -> int:
        return max([len(self.header), *(len(r) for r in self.rows)], default=0)


class TableIndexEntry(_Base):
    """1 dòng của `tables_index.jsonl` — chỉ text để BM25 index, không chứa CSV đầy đủ.

    CSV đầy đủ nằm trong per-doc cache (`tables_cache/`), đọc lazy khi cần.
    """

    table_ref: str
    doc_name: str
    ticker: str
    year: int | None = None
    scope: str | None = None
    line: int
    order: int = 0
    page: int | None = None
    n_rows: int = 0
    n_cols: int = 0
    index_text: str = ""


class RetrievalCandidate(_Base):
    table_ref: str
    doc_name: str
    ticker: str
    year: int | None = None
    scope: str | None = None
    line: int = 0
    page: int | None = None
    score: float = 0.0
    rank: int = 0
    context_before: str = ""
    csv_text: str = ""  # nhúng sẵn để Kaggle không cần mount corpus gốc


class RetrievalResult(_Base):
    """1 dòng của `retrieval_results.jsonl`."""

    id: int
    question: str
    tickers: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    scope: str | None = None
    candidates: list[RetrievalCandidate] = Field(default_factory=list)


class EvidenceItem(_Base):
    variable: str
    csv_path: str


class Prediction(_Base):
    """1 dòng của `predictions.jsonl` (ghi tăng dần trên Kaggle)."""

    id: int
    pandas_query: str = ""
    answer: float | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    used_table_refs: list[str] = Field(default_factory=list)
    exec_ok: bool = False
    exec_error: str | None = None
    raw_completion: str | None = None


class SubmissionItem(_Base):
    """Đúng schema `submission.json` ở COMPETITION.md mục 3."""

    id: int
    question: str
    answer: float
    relevant_docs: list[str]
    relevant_tables: list[str]
    evidence: list[EvidenceItem]
    pandas_query: str
