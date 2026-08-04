"""Gợi ý tiền tính sẵn (deterministic) nhúng vào prompt: cột nào là kỳ nào, đơn vị nào, đổi ra sao.

Vì sao cần (Bug 15): bài nộp V1 để LLM tự suy ra cột nào ứng với năm/kỳ được hỏi và tự đổi đơn vị.
Đo thật trên 1.012 câu: 56,9% câu dùng đúng một đoạn mẫu lấy **giá trị số cuối cùng** của dòng khớp
nhãn (`found[-1]`) — sai cột ở 83% mẫu đối chiếu tay được, và gần như không bao giờ đổi đơn vị
(chỉ 135/1.012 query có phép chia 1e6/1e9 trong khi ~694 câu hỏi nêu đơn vị triệu/tỷ/nghìn tỷ đồng).
Answer Accuracy thật: 1,38%.

Cách sửa: phần nào tính được bằng luật thì tính sẵn ở Python (rẻ, kiểm chứng được, không phụ thuộc
model 7B đoán đúng) rồi ghi thẳng vào prompt dưới dạng chú thích cột + hệ số đổi đơn vị:
- `column_hints()` — mỗi cột ứng với kỳ nào (ngày/năm/"số cuối năm"...) và đơn vị nào.
- `answer_unit()` — câu hỏi muốn đáp án theo đơn vị nào (hoặc là tỉ lệ/hệ số -> không đổi đơn vị).
- `target_column()` — cột khớp kỳ được hỏi (None nếu không chắc, khi đó prompt không gợi ý mù).
- `table_hint_lines()` — gói lại thành 1-2 dòng chú thích ngắn để nhúng vào prompt.

Module cố ý **không** đoán khi thiếu bằng chứng: gợi ý sai còn tệ hơn không gợi ý, vì model được
dặn tin theo chú thích này.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --- đơn vị -------------------------------------------------------------------------------------

# Hệ số quy về VND: giá trị trong bảng × factor = số VND.
#
# CỐ Ý không bỏ dấu trước khi so khớp đơn vị (khác phần nhận diện kỳ): bỏ dấu làm đơn vị đụng độ
# với từ thường gặp trong chính các bảng này — "đồng" ~ "động" (hoạt động, biến động, cổ đông),
# "tỷ" ~ "ty" (công ty), "ngàn" ~ "ngân" (ngân hàng). Đã gặp thật: cột "Hoạt động chính" bị nhận
# là đơn vị VND. Chấp nhận vài biến thể OCR phổ biến thay vì bỏ dấu toàn bộ.
# Thứ tự QUAN TRỌNG: cụm ghép ("trăm tỷ", "nghìn tỷ") phải xét trước bậc đơn ("tỷ").
# "trăm tỷ đồng" xuất hiện ở 66/1.012 câu hỏi (6,5%) — V1 hiểu thành "tỷ đồng" là sai hệ số 100 lần.
_SCALE_PATTERNS: tuple[tuple[float, str], ...] = (
    (1e12, r"(?:nghìn|nghin|ngàn)\s*t[ỷỉ]"),
    (1e11, r"trăm\s*t[ỷỉ]"),
    (1e10, r"chục\s*t[ỷỉ]"),
    (1e9, r"t[ỷỉ]"),
    (1e8, r"trăm\s*tri[ệê]u"),
    (1e7, r"chục\s*tri[ệê]u"),
    (1e6, r"tri[ệê]u|trieu"),
    (1e3, r"nghìn|nghin|ngàn"),
)
# `VND` không có `\b` ở đầu: header thật hay dính liền số ("31/12/2018VND"). Lookahead chặn
# "VNDIRECT". "đông" (cổ đông) KHÔNG được nhận là đồng — false positive đắt hơn OCR mất dấu.
_CURRENCY_RE = re.compile(r"VN[DĐ](?![A-Za-zÀ-ỹ])|đồng|dồng", re.IGNORECASE)
_UNIT_LABELS = {
    1.0: "VND",
    1e3: "Nghìn VND",
    1e6: "Triệu VND",
    1e7: "Chục triệu VND",
    1e8: "Trăm triệu VND",
    1e9: "Tỷ VND",
    1e10: "Chục tỷ VND",
    1e11: "Trăm tỷ VND",
    1e12: "Nghìn tỷ VND",
}


def _ascii_lower(text: str) -> str:
    """Bỏ dấu + hạ chữ để so khớp từ khoá bền với lỗi OCR dấu ("Trieu"/"Triệu")."""
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    return stripped.replace("đ", "d").replace("Đ", "D").lower()


def unit_factor(text: str) -> float | None:
    """Hệ số quy về VND của đơn vị nêu trong `text`, hoặc None nếu không thấy đơn vị tiền nào.

    "VND"/"đồng" -> 1, "triệu VND"/"triệu đồng" -> 1e6, "tỷ đồng" -> 1e9, "nghìn tỷ đồng" -> 1e12.
    Chỉ nhận khi có token tiền tệ (VND/đồng) để "3 triệu cổ phiếu" không bị hiểu là đơn vị tiền.
    """
    if not text:
        return None
    match = _CURRENCY_RE.search(text)
    if match is None:
        return None
    # Từ chỉ bậc (triệu/tỷ...) đứng NGAY TRƯỚC token tiền tệ ("triệu VND"), hoặc ngay sau khi OCR
    # đảo thứ tự ("VND triệu" — hiếm). Chỉ soi cửa sổ hẹp để không bắt "tỷ" của câu khác.
    window = text[max(0, match.start() - 24) : match.end() + 12]
    for factor, pattern in _SCALE_PATTERNS:
        if re.search(pattern, window, re.IGNORECASE):
            return factor
    return 1.0


def unit_label(factor: float | None) -> str | None:
    return _UNIT_LABELS.get(factor) if factor is not None else None


_PERCENT_RE = re.compile(r"%|phần\s*trăm", re.IGNORECASE)
_RATIO_WORDS = (
    "tỷ lệ",
    "tỉ lệ",
    "tỷ số",
    "tỉ số",
    "tỷ trọng",
    "hệ số",
    "biên lợi nhuận",
    "tốc độ tăng trưởng",
    "roe",
    "roa",
    "bao nhiêu lần",
    "gấp bao nhiêu",
)
# Đáp án KHÔNG phải số tiền: một năm ("năm nào ... cao nhất"), một số đếm ("bao nhiêu công ty"),
# hay số lượng cổ phiếu/cổ phần. ~8% câu thuộc nhóm này — nếu vẫn gợi ý đổi đơn vị tiền thì model
# sẽ chia 1e9 một con số vốn dĩ đã đúng.
_NON_MONETARY_RE = re.compile(
    r"năm nào|quý nào|thời điểm nào|công ty nào|doanh nghiệp nào|mã nào|đơn vị nào"
    r"|bao nhiêu năm|số năm|bao nhiêu công ty|bao nhiêu doanh nghiệp|bao nhiêu mã"
    r"|bao nhiêu ngân hàng|bao nhiêu tổ chức|tổng số công ty|tổng số doanh nghiệp|tổng số ngân hàng"
    r"|bao nhiêu cổ phiếu|số lượng cổ phiếu|số cổ phiếu|cổ phần nào|bao nhiêu cổ phần|số lượng cổ phần"
    r"|tổng số lượng cổ phần",
    re.IGNORECASE,
)
# Ngoại tệ: đáp án theo USD/EUR không quy đổi được sang VND bằng luật (không có tỷ giá).
_FOREIGN_CURRENCY_RE = re.compile(r"\bUSD\b|\bEUR\b|\bJPY\b|đô la|dollar", re.IGNORECASE)
# "nợ ... nhiều hơn ... mấy triệu thế nhỉ?" — bậc đơn vị nêu bằng khẩu ngữ, không kèm "đồng"/"VND".
_BARE_SCALE_ASK_RE = re.compile(
    rf"(?:bao nhiêu|mấy)\s+({'|'.join(pattern for _, pattern in _SCALE_PATTERNS)})\b",
    re.IGNORECASE,
)
# Dựng từ chính `_SCALE_PATTERNS` (đã xếp cụm ghép trước bậc đơn) để 2 chỗ không lệch nhau.
_UNIT_ASK_RE = re.compile(
    rf"({'|'.join(pattern for _, pattern in _SCALE_PATTERNS)})?\s*(đồng|VN[DĐ])(?![A-Za-zÀ-ỹ])",
    re.IGNORECASE,
)


_ASK_LABELS = {
    1.0: "đồng",
    1e3: "nghìn đồng",
    1e6: "triệu đồng",
    1e7: "chục triệu đồng",
    1e8: "trăm triệu đồng",
    1e9: "tỷ đồng",
    1e10: "chục tỷ đồng",
    1e11: "trăm tỷ đồng",
    1e12: "nghìn tỷ đồng",
}
_INTERROGATIVE_RE = re.compile(r"bao nhiêu|bằng bao nhiêu|là mấy|mấy trăm|theo đơn vị", re.IGNORECASE)
_COUNT_SCALE_RE = re.compile(r"(nghìn|triệu|t[ỷỉ])\s*(?:cổ phiếu|đơn vị)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AnswerUnit:
    """Đơn vị mà câu hỏi muốn đáp án trả về."""

    factor: float | None  # hệ số quy về VND (1e6 = "triệu đồng"); None = không xác định
    label: str | None  # "triệu đồng"...
    no_conversion: bool = False  # đáp án không phải số tiền (%, hệ số, năm, số đếm) -> không đổi đơn vị
    reason: str = ""  # vì sao không đổi đơn vị — đưa nguyên văn vào prompt
    assumed: bool = False  # câu hỏi không nêu đơn vị -> mặc định VND thô (theo ví dụ nộp bài của BTC)

    @property
    def known(self) -> bool:
        return self.factor is not None or self.no_conversion


def answer_unit(question: str) -> AnswerUnit:
    """Đơn vị đáp án suy từ câu hỏi — chỉ nhận cụm đơn vị nằm **sau** từ hỏi ("bao nhiêu ...").

    Vì sao phải sau từ hỏi: câu hỏi hay nhắc đơn vị ở mệnh đề mô tả ("Vốn điều lệ 5.000 tỷ đồng
    ... chiếm bao nhiêu phần trăm?") — cụm đó nói về dữ kiện, không phải đơn vị đáp án.

    Câu hỏi hỏi %/hệ số/lần: `is_ratio=True`, không đổi đơn vị tiền (tử và mẫu triệt tiêu).
    Câu hỏi không nêu đơn vị nào: mặc định VND thô (`assumed=True`) — khớp ví dụ submission của BTC
    ("Doanh thu thuần ... năm 2023 là bao nhiêu?" -> `63075000000`).
    """
    lowered = question.lower()
    asks = list(_INTERROGATIVE_RE.finditer(question))
    cutoff = asks[-1].start() if asks else 0
    matches = [m for m in _UNIT_ASK_RE.finditer(question) if m.start() >= cutoff]
    # `%` sau từ hỏi là bằng chứng mạnh nhất; nếu sau từ hỏi KHÔNG có đơn vị tiền nào thì `%` ở bất
    # kỳ đâu cũng tính ("Tốc độ tăng trưởng % ... là bao nhiêu?"). Ngược lại, có đơn vị tiền sau từ
    # hỏi mà `%` chỉ nằm ở mệnh đề dữ kiện ("chiếm 5% ... là bao nhiêu tỷ đồng?") thì tiền thắng.
    percent = bool(_PERCENT_RE.search(question[cutoff:])) or (
        not matches and bool(_PERCENT_RE.search(question))
    )
    ratio = percent or any(word in lowered for word in _RATIO_WORDS)
    if _NON_MONETARY_RE.search(question):
        reason = "đáp án không phải số tiền (một năm / số đếm / số lượng cổ phiếu)"
        count_scale = _COUNT_SCALE_RE.search(question)
        if count_scale:
            for value, pattern in _SCALE_PATTERNS:
                if re.fullmatch(pattern, count_scale.group(1), re.IGNORECASE):
                    reason += f" — nhưng câu hỏi hỏi theo '{count_scale.group(0)}', hãy chia {value:g}"
                    break
        return AnswerUnit(factor=None, label=None, no_conversion=True, reason=reason)
    if not matches:
        if ratio:
            return AnswerUnit(
                factor=None, label=None, no_conversion=True, reason="câu hỏi hỏi tỉ lệ/hệ số"
            )
        if _FOREIGN_CURRENCY_RE.search(question[cutoff:]):
            return AnswerUnit(
                factor=None,
                label=None,
                no_conversion=True,
                reason="câu hỏi hỏi theo ngoại tệ (USD/EUR…) — lấy đúng cột ngoại tệ, KHÔNG quy đổi VND",
            )
        bare = _BARE_SCALE_ASK_RE.search(question)  # regex đã tự chứa từ hỏi ("bao nhiêu"/"mấy")
        if bare:
            for value, pattern in _SCALE_PATTERNS:
                if re.fullmatch(pattern, bare.group(1).strip(), re.IGNORECASE):
                    return AnswerUnit(factor=value, label=_ASK_LABELS[value])
        return AnswerUnit(factor=1.0, label="đồng (VND thô)", assumed=True)
    if percent:  # "... bao nhiêu phần trăm" thắng cụm "tỷ đồng" đứng cùng mệnh đề
        return AnswerUnit(
            factor=None,
            label=None,
            no_conversion=True,
            reason="câu hỏi hỏi tỉ lệ phần trăm — trả về theo thang % (0,9 -> result = 90.0)",
        )
    scale, _currency = matches[-1].groups()
    factor = 1.0
    if scale:
        for value, pattern in _SCALE_PATTERNS:
            if re.fullmatch(pattern, scale.strip(), re.IGNORECASE):
                factor = value
                break
    return AnswerUnit(factor=factor, label=_ASK_LABELS[factor])


# --- kỳ báo cáo của từng cột --------------------------------------------------------------------

_DATE_RE = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*((?:19|20)\d{2})")
_MONTH_YEAR_RE = re.compile(r"(?<![\d/])(\d{1,2})\s*[/\-.]\s*((?:19|20)\d{2})(?![\d/])")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_VN_DATE_RE = re.compile(r"ngay\s*(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*((?:19|20)\d{2})")

# Nhãn kỳ tương đối (không có năm) -> khoá chuẩn hoá.
#
# Cố ý tách 2 HỌ nhãn khác nhau về ngữ nghĩa (phát hiện khi đối chiếu tay câu "Tiền ... đầu năm 2022"):
# - Số dư: "Số cuối năm"/"Số đầu năm" -> hai MỐC của cùng một niên độ (bảng cân đối, thuyết minh số dư).
# - Niên độ: "Năm nay"/"Năm trước" -> hai NĂM TÀI CHÍNH khác nhau (KQKD, lưu chuyển tiền tệ).
# Trộn 2 họ này là sai: câu "Tiền và tương đương tiền đầu năm 2022" trên báo cáo lưu chuyển tiền tệ có
# "đầu năm" nằm ở NHÃN DÒNG, cột cần lấy vẫn là "Năm nay" của report 2022 — không phải "Năm trước".
PERIOD_CLOSING = "cuối năm/cuối kỳ"
PERIOD_OPENING = "đầu năm/đầu kỳ"
PERIOD_CURRENT = "năm nay"
PERIOD_PREVIOUS = "năm trước"
_CLOSING_MARKERS = ("cuoi nam", "cuoi ky", "ending balance", "cuoi thang")
_OPENING_MARKERS = ("dau nam", "dau ky", "beginning balance", "dau thang")
_CURRENT_MARKERS = ("nam nay", "ky nay", "nam hien tai")
_PREVIOUS_MARKERS = ("nam truoc", "ky truoc")


@dataclass(frozen=True, slots=True)
class ColumnHint:
    """Một cột của bảng: nó thuộc kỳ nào, đơn vị nào."""

    index: int
    name: str
    period: str | None = None  # "31/12/2018" | "2018" | một trong 4 hằng `PERIOD_*`
    year: int | None = None  # năm suy được từ nhãn kỳ (nếu nhãn có năm)
    unit_factor: float | None = None
    is_percent: bool = False

    @property
    def unit_text(self) -> str | None:
        return "%" if self.is_percent else unit_label(self.unit_factor)


def _period_of(name: str) -> tuple[str | None, int | None]:
    flat = _ascii_lower(name)
    vn = _VN_DATE_RE.search(flat)
    if vn:
        day, month, year = vn.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}", int(year)
    date = _DATE_RE.search(flat)
    if date:
        day, month, year = date.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}", int(year)
    month_year = _MONTH_YEAR_RE.search(flat)
    if month_year:
        month, year = month_year.groups()
        return f"{int(month):02d}/{year}", int(year)
    year_only = _YEAR_RE.search(flat)
    if year_only:
        return year_only.group(1), int(year_only.group(1))
    if any(marker in flat for marker in _CLOSING_MARKERS):
        return PERIOD_CLOSING, None
    if any(marker in flat for marker in _OPENING_MARKERS):
        return PERIOD_OPENING, None
    if any(marker in flat for marker in _CURRENT_MARKERS):
        return PERIOD_CURRENT, None
    if any(marker in flat for marker in _PREVIOUS_MARKERS):
        return PERIOD_PREVIOUS, None
    return None, None


def column_hints(header: list[str]) -> list[ColumnHint]:
    """Suy kỳ + đơn vị cho từng cột từ tên cột (đã gộp header nhiều tầng — xem Bug 14)."""
    hints: list[ColumnHint] = []
    for i, name in enumerate(header):
        period, year = _period_of(name)
        flat = _ascii_lower(name)
        hints.append(
            ColumnHint(
                index=i,
                name=name,
                period=period,
                year=year,
                unit_factor=unit_factor(name),
                is_percent="%" in name and not _CURRENCY_RE.search(flat),
            )
        )
    return hints


def table_unit_factor(header: list[str], context_before: str = "") -> float | None:
    """Đơn vị chung của bảng: ưu tiên tên cột, sau đó cụm "Đơn vị (tính): ..." trong ngữ cảnh.

    KHÔNG quét đơn vị trong câu văn ngữ cảnh bất kỳ: câu kiểu "chi phí lãi vay được vốn hoá là
    31.729 triệu VND" nói về số trong CHÍNH câu đó, không phải đơn vị của bảng bên dưới.
    """
    factors = [f for f in (unit_factor(name) for name in header) if f is not None]
    if factors:
        return max(set(factors), key=factors.count)  # đơn vị xuất hiện ở nhiều cột nhất
    match = re.search(r"(đơn\s*vị|don\s*vi)[^.\n]{0,40}", context_before, re.IGNORECASE)
    return unit_factor(match.group(0)) if match else None


# --- khớp kỳ được hỏi với cột -------------------------------------------------------------------

_ASK_END_RE = re.compile(r"cuoi nam|den ngay|tai ngay|den (?:31|30)|thoi diem|cuoi ky|cuoi quy")
_ASK_START_RE = re.compile(r"dau nam|dau ky|dau thang|thoi diem dau")
# Cột phụ của cùng một kỳ: khi nhiều cột khớp kỳ như nhau, những cột này gần như chắc chắn KHÔNG
# phải cột giá trị chính (bảng hàng tồn kho: "Số cuối năm | Giá gốc" vs "Số cuối năm | Dự phòng").
_SECONDARY_COLUMN_RE = re.compile(
    r"dự phòng|du phong|hao mòn|hao mon|lũy kế|luy ke|khả năng trả nợ|kha nang tra no|giảm|giam",
    re.IGNORECASE,
)


def _prefer(indices: list[int], hints: list[ColumnHint]) -> int:
    """Chọn 1 trong các cột khớp kỳ như nhau: ưu tiên cột giá trị chính, sau đó cột đứng trước."""
    by_index = {h.index: h for h in hints}
    return min(
        indices,
        key=lambda i: (bool(_SECONDARY_COLUMN_RE.search(by_index[i].name if i in by_index else "")), i),
    )


@dataclass(frozen=True, slots=True)
class AskedPeriod:
    """Kỳ mà câu hỏi nhắm tới."""

    date: str | None = None  # "31/12/2016" nếu câu hỏi nêu ngày cụ thể
    year: int | None = None
    wants_start: bool = False  # "đầu năm N"
    wants_end: bool = False  # "cuối năm N" / "đến 31/12/N"


def asked_period(question: str, years: list[int] | None = None) -> AskedPeriod:
    flat = _ascii_lower(question)
    date = None
    year = None
    vn = _VN_DATE_RE.search(flat) or _DATE_RE.search(flat)
    if vn:
        day, month, y = vn.groups()
        date = f"{int(day):02d}/{int(month):02d}/{y}"
        year = int(y)
    if year is None:
        found = _YEAR_RE.findall(flat)
        if found:
            year = int(found[-1])
        elif years:
            year = max(years)
    wants_start = bool(_ASK_START_RE.search(flat))
    wants_end = bool(_ASK_END_RE.search(flat)) and not wants_start
    return AskedPeriod(date=date, year=year, wants_start=wants_start, wants_end=wants_end)


def target_column(
    hints: list[ColumnHint], asked: AskedPeriod, *, table_year: int | None = None
) -> int | None:
    """Cột khớp kỳ được hỏi, hoặc None nếu không đủ chắc.

    Thứ tự ưu tiên (dừng ở bước đầu tiên có kết quả duy nhất):
    1. Cột có đúng ngày câu hỏi nêu ("đến 31/12/2016" -> cột "31/12/2016").
    2. Cột có đúng năm được hỏi (và, nếu câu hỏi nói "đầu năm", cột ngày 01/01 của năm đó).
    3. Nhãn tương đối: report đúng năm được hỏi -> "cuối năm/năm nay" cho câu hỏi cuối năm,
       "đầu năm/năm trước" cho câu hỏi đầu năm; report năm N+1 -> "cuối năm N" nằm ở cột đầu năm.
    Cột phần trăm bị loại khỏi ứng viên khi câu hỏi hỏi số tiền (xử lý ở `table_hint_lines`).
    """
    if asked.date:
        exact = [h.index for h in hints if h.period == asked.date]
        if exact:
            return _prefer(exact, hints)

    if asked.year is not None:
        same_year = [h for h in hints if h.year == asked.year]
        if asked.wants_start:
            starts = [h.index for h in same_year if h.period and h.period.startswith("01/01")]
            if starts:
                return _prefer(starts, hints)
            # "đầu năm N" = "cuối năm N-1": bảng ghi ngày cụ thể thường không có cột 01/01/N,
            # mà có cột 31/12/(N-1) (cùng quy ước `year_expand` của bước retrieval).
            prev_end = [
                h.index
                for h in hints
                if h.year == asked.year - 1 and h.period and h.period.startswith(("31/12", "30/"))
            ]
            if prev_end:
                return _prefer(prev_end, hints)
        if asked.wants_end:
            ends = [h.index for h in same_year if h.period and h.period.startswith(("31/12", "30/"))]
            if ends:
                return _prefer(ends, hints)
            next_start = [
                h.index
                for h in hints
                if h.year == asked.year + 1 and h.period and h.period.startswith("01/01")
            ]
            if next_start:
                return _prefer(next_start, hints)
        if same_year and (len(same_year) == 1 or not (asked.wants_start or asked.wants_end)):
            return _prefer([h.index for h in same_year], hints)

    closing = [h.index for h in hints if h.period == PERIOD_CLOSING]
    opening = [h.index for h in hints if h.period == PERIOD_OPENING]
    current = [h.index for h in hints if h.period == PERIOD_CURRENT]
    previous = [h.index for h in hints if h.period == PERIOD_PREVIOUS]
    if table_year is not None and asked.year is not None:
        if asked.year == table_year:
            if asked.wants_start and opening:
                return _prefer(opening, hints)
            if asked.wants_end and closing:
                return _prefer(closing, hints)
            # Bảng chỉ có cột niên độ ("Năm nay"/"Năm trước"): "đầu năm N"/"cuối năm N" khi đó là
            # NHÃN DÒNG (báo cáo lưu chuyển tiền tệ, bảng biến động số dư), cột vẫn là niên độ N.
            if closing:
                return _prefer(closing, hints)
            if current:
                return _prefer(current, hints)
        if asked.year == table_year - 1:
            # "cuối năm N" = "đầu năm N+1" (số dư); với cột niên độ thì năm N là "Năm trước".
            if opening:
                return _prefer(opening, hints)
            if previous:
                return _prefer(previous, hints)
    return None


# --- gói lại thành chú thích cho prompt ---------------------------------------------------------


def _fmt_scale(factor: float) -> str:
    if factor >= 1:
        return f"× {factor:,.0f}".replace(",", ".")
    return f"÷ {1 / factor:,.0f}".replace(",", ".")


def scale_note(table_factor: float | None, answer: AnswerUnit) -> str:
    """Câu chú thích đổi đơn vị cho 1 bảng (đã biết đơn vị bảng + đơn vị đáp án)."""
    if answer.no_conversion:
        return f"Đổi đơn vị: KHÔNG đổi ({answer.reason})."
    if table_factor is None or answer.factor is None:
        return (
            "Đổi đơn vị: chưa suy được đơn vị của bảng từ header/ngữ cảnh — tự đọc đơn vị của bảng"
            f" rồi đổi sang đơn vị câu hỏi{f' ({answer.label})' if answer.label else ''}."
        )
    ratio = table_factor / answer.factor
    if ratio == 1:
        return f"Đổi đơn vị: KHÔNG cần — bảng đã ở {unit_label(table_factor)}, câu hỏi hỏi {answer.label}."
    return (
        f"Đổi đơn vị: số trong bảng đang là {unit_label(table_factor)}, câu hỏi hỏi {answer.label}"
        f" -> {_fmt_scale(ratio)} (nhân với {ratio:g}) TRƯỚC khi gán vào `result`."
    )


def column_note(hints: list[ColumnHint], target: int | None) -> str:
    """Chú thích 1 dòng: mỗi cột thuộc kỳ nào/đơn vị nào + cột nào khớp kỳ được hỏi.

    Cố ý ngắn (ngân sách prompt 16.000 ký tự cho tối đa 6 bảng): chỉ ghi cột suy được kỳ hoặc đơn vị.
    """
    parts: list[str] = []
    for hint in hints:
        bits = []
        if hint.period:
            bits.append(f"kỳ {hint.period}")
        if hint.unit_text:
            bits.append(f"({hint.unit_text})")
        if bits:
            parts.append(f"{hint.index}=" + " ".join(bits))
    listing = " ".join(parts) if parts else "(header không nêu kỳ)"
    if target is not None:
        pick = f"CỘT ỨNG VỚI KỲ ĐƯỢC HỎI -> .iloc[:, {target}]"
    else:
        pick = "cột ứng với kỳ được hỏi: chưa xác định được -> tự đối chiếu header với câu hỏi"
    return f"Cột theo kỳ: {listing} | {pick}"


def table_hint_lines(
    header: list[str],
    context_before: str,
    *,
    question: str,
    years: list[int] | None = None,
    table_year: int | None = None,
) -> tuple[str, int | None]:
    """2 dòng chú thích nhúng dưới tag `<table>` + chỉ số cột gợi ý (None nếu không đủ chắc)."""
    hints = column_hints(header)
    answer = answer_unit(question)
    asked = asked_period(question, years)
    # Câu hỏi so sánh nhiều năm ("trong các năm 2017, 2018 và 2020..."): với ĐÚNG bảng này, năm cần
    # lấy là năm của chính report — bảng 2018 chỉ có số 2018 (và 2017 ở cột đầu năm). Không ép theo
    # năm cuối cùng nhắc trong câu, nếu không mọi bảng của câu so sánh đều bị gợi ý sai cột.
    if table_year is not None and years and len(years) > 1 and table_year in years:
        asked = AskedPeriod(
            date=asked.date if asked.date and str(table_year) in asked.date else None,
            year=table_year,
            wants_start=asked.wants_start,
            wants_end=asked.wants_end,
        )
    # Câu hỏi số tiền thì cột "%" không thể là cột đáp án -> loại khỏi ứng viên khớp kỳ.
    candidates = (
        [h for h in hints if not h.is_percent]
        if not answer.no_conversion and any(h.is_percent for h in hints)
        else hints
    )
    target = target_column(candidates, asked, table_year=table_year)
    table_factor = table_unit_factor(header, context_before)
    if target is not None:
        # Đơn vị có thể khác nhau theo cột ("31/12/2018 | Triệu đồng" vs "31/12/2018 | %") — nếu biết
        # cột đích thì lấy đơn vị của chính cột đó thay cho đơn vị chung của bảng.
        column_factor = next((h.unit_factor for h in hints if h.index == target), None)
        if column_factor is not None:
            table_factor = column_factor
    return f"{column_note(hints, target)}\n{scale_note(table_factor, answer)}", target
