"""Calculate refund windows from the current policy evidence."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re

from retriever import RetrievalHit

_DATE_PATTERN = re.compile(
    r"\b(?P<day>\d{1,2})\s+"
    r"(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)"
    r"(?:\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_BUSINESS_RANGE_PATTERN = re.compile(
    r"(?P<minimum>\d+)\s+to\s+(?P<maximum>\d+)\s+business days?",
    re.IGNORECASE,
)
_PAYMENT_METHOD_PATTERNS = (
    ("bank transfer (NEFT) for cash-on-delivery orders", (
        r"\b(?:cash[\s-]*on[\s-]*delivery|cod)\b.*\b(?:bank transfer|neft)\b",
        r"\b(?:bank transfer|neft)\b.*\b(?:cash[\s-]*on[\s-]*delivery|cod)\b",
    )),
    ("ZentroMart Wallet for cash-on-delivery orders", (
        r"\b(?:cash[\s-]*on[\s-]*delivery|cod)\b.*\bwallet\b",
        r"\bwallet\b.*\b(?:cash[\s-]*on[\s-]*delivery|cod)\b",
    )),
    ("UPI", (r"\bupi\b",)),
    ("Credit or debit card", (
        r"\bcredit\s+card\b",
        r"\bdebit\s+card\b",
        r"\b(?:credit|debit)\s+(?:or|and)\s+(?:credit|debit)\s+card\b",
    )),
    ("Net banking", (r"\bnet[\s-]*banking\b",)),
    ("ZentroMart Wallet", (r"\bzentromart\s+wallet\b", r"\bwallet\b")),
)
_REFUND_QUESTION_PATTERN = re.compile(
    r"\brefund(?:s|ed|ing)?\b|\bmoney\s+back\b|\bget\s+my\s+money\b|"
    r"रिफ़?ंड|वापस\s+(?:पैसे|रिफ़?ंड)",
    re.IGNORECASE,
)
_COD_PATTERN = re.compile(
    r"\bcash[\s-]*on[\s-]*delivery\b|\bcod\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RefundWindow:
    payment_method: str
    initiation_date: date
    earliest_date: date
    latest_date: date


def _section_text(evidence: list[RetrievalHit], section_id: str) -> str | None:
    for hit in evidence:
        if hit.metadata.get("section_id") == section_id:
            return hit.text
    return None


def _payment_method(question: str) -> str | None:
    folded = question.casefold()
    for canonical, patterns in _PAYMENT_METHOD_PATTERNS:
        if any(re.search(pattern, folded) for pattern in patterns):
            return canonical
    return None


def _policy_timeline(refund_policy: str, payment_method: str) -> tuple[int, int] | None:
    canonical_method = re.sub(r"[^a-z0-9]+", " ", payment_method.casefold()).strip()
    for line in refund_policy.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip().strip("*`").casefold() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        policy_method = re.sub(r"[^a-z0-9]+", " ", cells[0]).strip()
        if policy_method != canonical_method:
            continue
        match = _BUSINESS_RANGE_PATTERN.search(cells[1])
        if match:
            return int(match.group("minimum")), int(match.group("maximum"))
    return None


def _cod_option_timelines(refund_policy: str) -> tuple[str, str] | None:
    wallet_timeline = None
    bank_timeline = None
    for line in refund_policy.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [
            cell.strip().strip("*`")
            for cell in line.strip().strip("|").split("|")
        ]
        if len(cells) < 2:
            continue
        method = re.sub(r"[^a-z0-9]+", " ", cells[0].casefold()).strip()
        if "cash on delivery orders" not in method:
            continue
        if "zentromart wallet" in method:
            wallet_timeline = cells[1]
        elif "bank transfer" in method:
            bank_timeline = cells[1]
    if wallet_timeline and bank_timeline:
        return wallet_timeline, bank_timeline
    return None


def format_cod_refund_options(
    question: str,
    evidence: list[RetrievalHit],
    reply_language: str,
) -> str | None:
    """Format COD options only when REF-2 and REF-3 establish both choices."""
    if (
        not _REFUND_QUESTION_PATTERN.search(question)
        or not _COD_PATTERN.search(question)
    ):
        return None
    refund_policy = _section_text(evidence, "REF-2")
    cod_policy = _section_text(evidence, "REF-3")
    if not refund_policy or not cod_policy:
        return None
    cod_policy_folded = cod_policy.casefold()
    if (
        "cash-on-delivery" not in cod_policy_folded
        or "verified" not in cod_policy_folded
        or "bank transfer" not in cod_policy_folded
        or "wallet" not in cod_policy_folded
    ):
        return None
    timelines = _cod_option_timelines(refund_policy)
    if timelines is None:
        return None
    wallet_timeline, bank_timeline = timelines

    if reply_language == "hi":
        return (
            "नमस्ते, मैं ZentroMart का AI ग्राहक-सहायता सहायक हूँ। COD रिफ़ंड "
            "के लिए आप ZentroMart Wallet चुन सकते हैं "
            f"({wallet_timeline}) या बैंक ट्रांसफ़र। बैंक डिटेल्स सत्यापित "
            "होने के बाद बैंक ट्रांसफ़र शुरू होता है और इसमें "
            f"{bank_timeline} लगते हैं [REF-3] [REF-2]।"
        )
    if reply_language == "hi-en":
        return (
            "Namaste, main ZentroMart ka AI customer-support assistant hoon. "
            "COD refund ke liye aap ZentroMart Wallet "
            f"({wallet_timeline}) ya bank transfer chun sakte hain. Bank "
            "details verify hone ke baad bank transfer shuru hota hai aur "
            f"{bank_timeline} lagte hain [REF-3] [REF-2]."
        )
    return (
        "Hello! I am ZentroMart's AI customer-support assistant. For a "
        "cash-on-delivery refund, you can choose ZentroMart Wallet "
        f"({wallet_timeline}) or a bank transfer. The bank transfer starts "
        "only after your bank details are verified and then takes "
        f"{bank_timeline} [REF-3] [REF-2]."
    )


def _initiation_date(question: str, default_year: int) -> date | None:
    iso_match = _ISO_DATE_PATTERN.search(question)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group())
        except ValueError:
            return None
    match = _DATE_PATTERN.search(question)
    if not match:
        return None
    year = int(match.group("year")) if match.group("year") else default_year
    for date_format in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(
                f"{match.group('day')} {match.group('month')} {year}",
                date_format,
            ).date()
        except ValueError:
            continue
    return None


def _frozen_reference_year(date_context: str) -> int | None:
    match = _DATE_PATTERN.search(date_context)
    if not match:
        return None
    year = match.groupdict().get("year")
    return int(year) if year else None


def _holiday_dates(business_day_context: str) -> set[date]:
    holidays = set()
    for value in _ISO_DATE_PATTERN.findall(business_day_context):
        holidays.add(date.fromisoformat(value))
    return holidays


def _add_business_days(start: date, count: int, holidays: set[date]) -> date:
    current = start
    completed = 0
    while completed < count:
        current += timedelta(days=1)
        if current.weekday() != 6 and current not in holidays:
            completed += 1
    return current


def calculate_refund_window(
    question: str,
    evidence: list[RetrievalHit],
) -> RefundWindow | None:
    """Return a dated window only when both current policy sections are present."""
    if not _REFUND_QUESTION_PATTERN.search(question):
        return None
    method = _payment_method(question)
    refund_policy = _section_text(evidence, "REF-2")
    business_day_context = _section_text(evidence, "KB-00.3")
    if not method or not refund_policy or not business_day_context:
        return None

    timeline = _policy_timeline(refund_policy, method)
    if timeline is None:
        return None
    reference_date = _section_text(evidence, "KB-00.2")
    default_year = (
        _frozen_reference_year(reference_date)
        if reference_date is not None
        else None
    )
    if default_year is None:
        return None
    initiation_date = _initiation_date(question, default_year)
    if initiation_date is None:
        return None

    holidays = _holiday_dates(business_day_context)
    minimum, maximum = timeline
    if minimum < 1 or maximum < minimum:
        return None
    return RefundWindow(
        payment_method=method,
        initiation_date=initiation_date,
        earliest_date=_add_business_days(initiation_date, minimum, holidays),
        latest_date=_add_business_days(initiation_date, maximum, holidays),
    )


def format_refund_window(window: RefundWindow, reply_language: str) -> str:
    """Render a deterministic, cited date answer in a supported dataset language."""
    start = window.initiation_date.strftime("%d %B %Y")
    earliest = window.earliest_date.strftime("%d %B %Y")
    latest = window.latest_date.strftime("%d %B %Y")

    if reply_language == "hi":
        return (
            "नमस्ते, मैं ZentroMart का AI ग्राहक-सहायता सहायक हूँ।\n\n"
            f"रिफ़ंड नीति के अनुसार, {start} को शुरू किए गए "
            f"{window.payment_method} रिफ़ंड की अनुमानित अवधि {earliest} "
            f"से {latest} तक है। गिनती अगले पात्र व्यावसायिक दिन से शुरू "
            "होती है; रविवार और सूचीबद्ध वेयरहाउस छुट्टियाँ नहीं गिनी जातीं "
            "[REF-2] [KB-00.3]।"
        )
    if reply_language == "hi-en":
        return (
            "Namaste, main ZentroMart ka AI customer-support assistant hoon.\n\n"
            f"Refund policy ke mutabik, {start} ko shuru hua "
            f"{window.payment_method} refund {earliest} se {latest} tak "
            "aa sakta hai. Counting agle eligible business day se shuru "
            "hoti hai; Sunday aur listed warehouse holidays count nahi hote "
            "[REF-2] [KB-00.3]."
        )
    return (
        "Hello! I am ZentroMart's AI customer-support assistant.\n\n"
        f"Under the refund policy, a {window.payment_method} refund initiated "
        f"on {start} is expected between {earliest} and {latest}, inclusive. "
        "Business-day counting starts on the next eligible business day and "
        "excludes Sundays and listed warehouse holidays [REF-2] [KB-00.3]."
    )
