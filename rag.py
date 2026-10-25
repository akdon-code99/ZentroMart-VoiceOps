"""Evidence-grounded single-turn answers for the RAG evaluation stage."""

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

from llm_client import chat
from refund_timeline import (
    calculate_refund_window,
    format_cod_refund_options,
    format_refund_window,
)
from retriever import EvidenceRetriever, RetrievalHit


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    evidence: list[RetrievalHit]
    required_action: str = ""
    action_status: str = "not_required"
    grounding_status: str = "model_generated"
    decision: str = "answer"


_RETURN_PATTERN = re.compile(
    r"\breturn(?:s|ed|ing)?\b|\breturnable\b|\bsend it back\b|"
    r"\bbring it back\b|रिटर्न|वापस|लौटा|लौटाने",
    re.IGNORECASE,
)
_CONDITION_PATTERN = re.compile(
    r"\bopen(?:ed)?\b|\bunopened\b|\bseal(?:ed)?\b|\bused\b|"
    r"\bcondition\b|सील|खोल|खुला",
    re.IGNORECASE,
)
_RETURN_WINDOW_PATTERN = re.compile(
    r"\bhow\s+long\b|\bhow\s+many\s+days?\b|\breturn\s+(?:window|period|within)\b|"
    r"\b(?:days?|period)\s+(?:to|for)\s+return\b|\breturnable\b|"
    r"कितने\s+दिन|कितना\s+समय|रिटर्न\s+की\s+अवधि",
    re.IGNORECASE,
)
_FESTIVE_SALE_PATTERN = re.compile(
    r"\bfestive\b|\bsale\b|त्योहार|सेल",
    re.IGNORECASE,
)
_WARRANTY_PATTERN = re.compile(
    r"\bwarrant(?:y|ies)\b|\bassured\b|वारंटी|गारंटी",
    re.IGNORECASE,
)
_DAMAGE_PATTERN = re.compile(
    r"\bdamag(?:e|ed|ing)\b|\bcrack(?:ed)?\b|\bbroken\b|\btorn\b|"
    r"\bdefective\b|\bwrong item\b|\bwrong product\b|खराब|टूटा|फटा",
    re.IGNORECASE,
)
_REFUND_PATTERN = re.compile(
    r"\brefund(?:s|ed|ing)?\b|\bmoney\s+back\b|\bget\s+my\s+money\b|"
    r"रिफ़?ंड|वापस\s+(?:पैसे|रिफ़?ंड)",
    re.IGNORECASE,
)
_REFUND_TIMELINE_PATTERN = re.compile(
    r"\bhow\s+(?:long|fast|soon|quickly)\b|\bwhen\b|"
    r"\brefund\s+timeline\b|\b(?:business|working)\s+days?\b|"
    r"\bdays?\s+(?:for|until|to)\s+(?:the\s+)?refund\b|"
    r"कितने\s+दिन|कब\s+तक|कितनी\s+देर",
    re.IGNORECASE,
)
_REFUND_METHOD_PATTERN = re.compile(
    r"\bupi\b|\bnet[\s-]*banking\b|\bcredit\s+card\b|\bdebit\s+card\b|"
    r"\bzentromart\s+wallet\b|\bcash[\s-]*on[\s-]*delivery\b|\bcod\b",
    re.IGNORECASE,
)
_COD_PATTERN = re.compile(r"\bcash[\s-]*on[\s-]*delivery\b|\bcod\b", re.IGNORECASE)
_DATED_QUESTION_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
    re.IGNORECASE,
)
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_SKU_PATTERN = re.compile(r"\b[A-Z]{2,5}-\d{4}\b", re.IGNORECASE)
_PLUS_PATTERN = re.compile(
    r"\bzentromart\s+plus\b|\bplus\s+member\b",
    re.IGNORECASE,
)
_UNRECOGNIZED_ORDER_PATTERN = re.compile(
    r"\b(?:never|did not|didn't|have not|haven't)\s+"
    r"(?:place(?:d)?|order(?:ed)?|make)\b|"
    r"\b(?:order|payment|account activity)\s+i\s+did(?: not|n't)\b|"
    r"\bunauthori[sz]ed\b|\bnot my order\b",
    re.IGNORECASE,
)
_ACTION_CLAIM_PATTERN = re.compile(
    r"\b(?:creating|created|raising|raised|opening|opened)\s+(?:a\s+)?ticket\b|"
    r"\b(?:connecting|connected|transferring|transferred)\s+you\b|"
    r"\b(?:i(?:'ll| will)\s+(?:create|raise|open)\s+(?:a\s+)?ticket)\b|"
    r"\b(?:i(?:'ll| will)\s+connect|please hold while i connect)\b|"
    r"\b(?:i(?:'ve| have)\s+)?(?:escalated|handed over)\b",
    re.IGNORECASE,
)
_REFUSAL_PATTERN = re.compile(
    r"\b(?:not eligible|not allowed|not permitted|cannot approve|can't approve|"
    r"cannot provide|can't provide|cannot refund|can't refund|cannot cancel|"
    r"can't cancel|not entitled|policy does not allow|policy doesn't allow|"
    r"(?:cannot|can't|unable to)\s+(?:process|fulfil|fulfill|allow|grant|"
    r"honou?r|comply|do that|issue|accept)\b|not possible to\b|"
    r"अनुमति नहीं|पात्र नहीं|स्वीकृत नहीं कर सकता|रिफंड नहीं कर सकता|"
    r"नहीं कर सकता)",
    re.IGNORECASE,
)
_PICKUP_FEE_PATTERN = re.compile(
    r"\b(?:reverse|pickup|collection)\s+(?:pickup\s+)?(?:fee|charge)\b|"
    r"\bfee\b.{0,40}\b(?:pickup|collection)\b|"
    r"\b(?:pay|charged)\b.{0,40}\b(?:pickup|collection)\b|"
    r"\b(?:pickup|collection)\b.{0,40}\b(?:pay|charged|fee)\b|"
    r"\b(?:pickup|collection)\s+(?:is\s+)?free\b|"
    r"\bfree\s+(?:reverse\s+)?(?:pickup|collection)\b|"
    r"पिकअप.{0,12}(?:फ़ीस|फीस|शुल्क)",
    re.IGNORECASE,
)
_SERVICEABILITY_PATTERN = re.compile(
    r"\b(?:serviceable|delivery|deliver|ship)\b.{0,40}\b(?:to|in)\b",
    re.IGNORECASE,
)
_DAMAGE_RESOLUTION_PATTERN = re.compile(
    r"\brefund(?:s|ed|ing)?\b|\breplac(?:e|ed|ement)\b|\brepair(?:ed)?\b|"
    r"\bresolve\b|\bresolution\b|\bremedy\b|रिफ़?ंड|बदलाव|मरम्मत",
    re.IGNORECASE,
)
_DAMAGE_REPORT_QUESTION_PATTERN = re.compile(
    r"\bwhat\s+(?:should|can|do)\s+i\s+do\b|\bwhat\s+to\s+do\b|"
    r"\bhow\s+(?:do|can|should)\s+i\s+(?:report|complain)\b|"
    r"\bfile\s+(?:a\s+)?complaint\b|\breport\s+(?:it|the\s+damage)\b|"
    r"क्या\s+करूँ|क्या\s+करना\s+चाहिए|शिकायत|रिपोर्ट",
    re.IGNORECASE,
)
_DAMAGE_REPORT_ALREADY_FILED_PATTERN = re.compile(
    r"\balready\s+(?:filed|reported|lodged|submitted)\b.{0,60}"
    r"\b(?:report|complaint|damage|issue)\b|"
    r"\b(?:i|we)\s+(?:have\s+)?(?:already\s+)?"
    r"(?:filed|reported|lodged|submitted)\b.{0,60}"
    r"\b(?:report|complaint|damage|issue)\b|"
    r"\b(?:report|complaint|issue)\b.{0,40}"
    r"\b(?:already\s+)?(?:filed|reported|lodged|submitted)\b|"
    r"\b(?:filed|reported|lodged|submitted)\b.{0,60}"
    r"\b(?:report|complaint|damage|issue)\b.{0,30}\balready\b",
    re.IGNORECASE,
)
_REPORTING_WINDOW_SIGNAL_PATTERN = re.compile(
    r"\bwithin\b[^.\n]{0,60}\bhours?\b",
    re.IGNORECASE,
)
_REPORTING_WINDOW_PATTERN = re.compile(
    r"\bwithin\s+\*{0,2}(?P<hours>\d+)\s+hours?\*{0,2}"
    r"(?:\s+of\s+\*{0,2}(?P<anchor>delivery|purchase)\b"
    r"(?!\s*(?:or|and|/)\s*(?:delivery|purchase)\b))?",
    re.IGNORECASE,
)
_ITEM_AND_PACKAGING_PHOTOS_PATTERN = re.compile(
    r"\bwith\s+photos?\s+of\s+(?:the\s+)?item\b"
    r"[^.\n]{0,60}\b(?:and|&)\s+(?:the\s+)?packaging\b",
    re.IGNORECASE,
)
_PHOTO_REQUIREMENT_NEGATION_PATTERN = re.compile(
    r"\b(?:optional|without|not\s+(?:required|needed|necessary))\b",
    re.IGNORECASE,
)


def _section_ids(evidence: list[RetrievalHit]) -> set[str]:
    return {
        str(hit.metadata["section_id"])
        for hit in evidence
        if hit.metadata.get("section_id")
    }


def _product_rows(evidence: list[RetrievalHit]) -> list[tuple[RetrievalHit, dict[str, Any]]]:
    return [
        (hit, hit.metadata["row"])
        for hit in evidence
        if hit.metadata.get("source_type") == "csv"
        and hit.metadata.get("csv_table") == "products"
        and isinstance(hit.metadata.get("row"), dict)
    ]


def _matching_product_rows(
    question: str,
    evidence: list[RetrievalHit],
) -> list[tuple[RetrievalHit, dict[str, Any]]]:
    rows = _product_rows(evidence)
    skus = {sku.upper() for sku in _SKU_PATTERN.findall(question)}
    if skus:
        exact = [
            pair for pair in rows
            if str(pair[1].get("sku", "")).upper() in skus
        ]
        if exact:
            return exact
    folded_question = question.casefold()
    exact_titles = [
        pair for pair in rows
        if str(pair[1].get("title", "")).casefold() in folded_question
    ]
    return exact_titles or rows


def _has_explicit_product_reference(
    question: str,
    evidence: list[RetrievalHit],
) -> bool:
    skus = {sku.upper() for sku in _SKU_PATTERN.findall(question)}
    if skus:
        return any(
            str(row.get("sku", "")).upper() in skus
            for _, row in _product_rows(evidence)
        )
    folded_question = question.casefold()
    return any(
        str(row.get("title", "")).casefold() in folded_question
        for _, row in _product_rows(evidence)
    )


def _high_value_damage_product(
    question: str,
    evidence: list[RetrievalHit],
) -> tuple[RetrievalHit, dict[str, Any]] | None:
    if not _DAMAGE_PATTERN.search(question):
        return None
    for hit, row in _matching_product_rows(question, evidence):
        try:
            price = int(str(row.get("price_inr", "")).replace(",", ""))
        except ValueError:
            continue
        if price > 10000:
            return hit, row
    return None


def _missing_required_sections(
    question: str,
    evidence: list[RetrievalHit],
) -> set[str]:
    sections = _section_ids(evidence)
    required: set[str] = set()
    if _REFUND_PATTERN.search(question):
        if _COD_PATTERN.search(question):
            required.update(("REF-2", "REF-3"))
        elif (
            _REFUND_TIMELINE_PATTERN.search(question)
            or _REFUND_METHOD_PATTERN.search(question)
        ):
            required.add("REF-2")
        if _DATED_QUESTION_PATTERN.search(question):
            required.update(("KB-00.2", "KB-00.3"))
    if _RETURN_PATTERN.search(question):
        historical = any(
            hit.metadata.get("retrieval_role") == "historical_reference"
            for hit in evidence
        )
        if not historical:
            required.add("RET-2")
        if _CONDITION_PATTERN.search(question):
            required.add("RET-4")
    if _WARRANTY_PATTERN.search(question):
        required.add("WAR-1")
        if any(
            str(row.get("assured", "")).casefold() == "true"
            for _, row in _matching_product_rows(question, evidence)
        ):
            required.add("WAR-2")
    if _DAMAGE_PATTERN.search(question):
        required.update(("DWM-1", "DWM-2"))
        if _PICKUP_FEE_PATTERN.search(question):
            required.add("RET-6")
        if _high_value_damage_product(question, evidence):
            required.update(("DWM-3", "ESC-1", "ESC-2"))
    if _UNRECOGNIZED_ORDER_PATTERN.search(question):
        required.update(("ESC-1", "ESC-2"))
    return required - sections


def _safe_uncertainty(reply_language: str) -> str:
    if reply_language == "hi":
        return (
            "नमस्ते, मैं ZentroMart का AI ग्राहक-सहायता सहायक हूँ। "
            "उपलब्ध नीति-साक्ष्य से मैं इसकी पुष्टि नहीं कर सकता। "
            "आप किस खास जानकारी की पुष्टि चाहते हैं?"
        )
    if reply_language == "hi-en":
        return (
            "Namaste, main ZentroMart ka AI customer-support assistant hoon. "
            "Available policy evidence se main iski pushti nahi kar sakta. "
            "Aap kis specific detail ki pushti chahte hain?"
        )
    return (
        "Hello! I am ZentroMart's AI customer-support assistant. "
        "I cannot confirm this from the available policy evidence. "
        "Which specific detail would you like me to check?"
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _parse_generated_response(content: Any) -> tuple[str, str | None]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model response must be a non-empty JSON string.")

    text = content.strip()
    if text.startswith("```"):
        # Remove starting fence
        if text.startswith("```json"):
            text = text[7:]
        else:
            text = text[3:]
        # Remove trailing fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    payload = json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")

    kind = payload.get("kind")
    if kind == "abstention" and set(payload) == {"kind"}:
        return kind, None
    if kind == "factual_answer" and "answer" in payload:
        answer = payload["answer"]
        if isinstance(answer, str) and answer.strip():
            return kind, answer.strip()
    raise ValueError("Model response does not match the response schema.")


def _is_damage_reporting_request(question: str) -> bool:
    return (
        bool(_DAMAGE_PATTERN.search(question))
        and bool(_DAMAGE_REPORT_QUESTION_PATTERN.search(question))
        and not _DAMAGE_RESOLUTION_PATTERN.search(question)
        and not _PICKUP_FEE_PATTERN.search(question)
        and not _DAMAGE_REPORT_ALREADY_FILED_PATTERN.search(question)
    )


def _damage_reporting_answer(
    question: str,
    reply_language: str,
    evidence: list[RetrievalHit],
) -> str | None:
    if not _is_damage_reporting_request(question):
        return None

    policies = [
        hit
        for hit in evidence
        if hit.metadata.get("section_id") == "DWM-1"
        and hit.metadata.get("status") == "current"
        and hit.metadata.get("retrieval_role") != "historical_reference"
    ]
    if not policies:
        return None

    policy_text = "\n".join(dict.fromkeys(hit.text for hit in policies))
    window_signals = list(_REPORTING_WINDOW_SIGNAL_PATTERN.finditer(policy_text))
    windows = list(_REPORTING_WINDOW_PATTERN.finditer(policy_text))
    photo_requirements = list(
        _ITEM_AND_PACKAGING_PHOTOS_PATTERN.finditer(policy_text)
    )
    if (
        len(window_signals) != 1
        or len(windows) != 1
        or windows[0].group("anchor") is None
        or windows[0].group("anchor").casefold() != "delivery"
        or len(photo_requirements) != 1
    ):
        return None
    photo_requirement = photo_requirements[0]
    sentence_start = max(
        policy_text.rfind(separator, 0, photo_requirement.start())
        for separator in (".", "!", "?", "\n")
    ) + 1
    sentence_ends = [
        position
        for separator in (".", "!", "?", "\n")
        if (position := policy_text.find(separator, photo_requirement.end())) != -1
    ]
    sentence_end = min(sentence_ends, default=len(policy_text))
    photo_context = policy_text[sentence_start:sentence_end]
    if _PHOTO_REQUIREMENT_NEGATION_PATTERN.search(photo_context):
        return None

    window = windows[0].group("hours")
    if reply_language == "hi":
        return (
            f"नमस्ते, मैं ZentroMart का AI ग्राहक-सहायता सहायक हूँ। "
            f"डिलीवरी के {window} घंटे के भीतर आइटम और पैकेजिंग की फ़ोटो के "
            "साथ शिकायत करें [DWM-1]।"
        )
    if reply_language == "hi-en":
        return (
            f"Namaste, main ZentroMart ka AI customer-support assistant hoon. "
            f"Delivery ke {window} ghante ke andar item aur packaging ki "
            "photos ke saath complaint raise karein [DWM-1]."
        )
    return (
        "Hello! I am ZentroMart's AI customer-support assistant. "
        f"Report the issue within {window} hours of delivery, with photos of "
        "the item and packaging "
        "[DWM-1]."
    )


def _high_value_damage_answer(
    reply_language: str,
    product_hit: RetrievalHit,
) -> str:
    product_key = (
        f"products.csv:{product_hit.metadata.get('record_key')}"
    )
    if reply_language == "hi":
        return (
            "नमस्ते, मैं ZentroMart का AI ग्राहक-सहायता सहायक हूँ। "
            "₹10,000 से अधिक मूल्य की क्षतिग्रस्त वस्तु Priority Resolution "
            "टीम को भेजी जानी चाहिए और handoff से पहले टिकट बनाना आवश्यक है। "
            "इस RAG-only प्रवाह में टिकट बनाने की सुविधा नहीं है, इसलिए कोई "
            f"टिकट नहीं बनाया गया है [DWM-3] [ESC-1] [ESC-2] [{product_key}]।"
        )
    if reply_language == "hi-en":
        return (
            "Namaste, main ZentroMart ka AI customer-support assistant hoon. "
            "₹10,000 se zyada ke damaged item ke liye Priority Resolution "
            "team ko handoff aur usse pehle ticket zaroori hai. Is RAG-only "
            "flow mein ticket create karne ka tool connected nahi hai, isliye "
            f"koi ticket create nahi hua [DWM-3] [ESC-1] [ESC-2] [{product_key}]."
        )
    return (
        "Hello! I am ZentroMart's AI customer-support assistant. "
        "Because this damaged item is worth more than ₹10,000, policy requires "
        "a ticket before handoff to the Priority Resolution team. Ticket "
        "creation is not connected in this RAG-only flow, so no ticket has "
        f"been created [DWM-3] [ESC-1] [ESC-2] [{product_key}]."
    )


def _unexecuted_handoff_answer(
    reply_language: str,
    evidence: list[RetrievalHit],
) -> str:
    sections = _section_ids(evidence)
    citations = " ".join(
        f"[{section_id}]"
        for section_id in ("ESC-1", "ESC-2")
        if section_id in sections
    )
    if reply_language == "hi":
        return (
        "नमस्ते, मैं ZentroMart का AI ग्राहक-सहायता सहायक हूँ। इस मामले "
        "की समीक्षा किसी सहकर्मी को करनी चाहिए। इस चैट में टिकट नहीं "
        f"बनाया गया और handoff पूरा नहीं हुआ {citations}।"
        ).strip()
    if reply_language == "hi-en":
        return (
        "Namaste, main ZentroMart ka AI customer-support assistant hoon. "
        "Is case ko ek colleague ko review karna chahiye. Is chat mein "
        f"ticket create nahi hua aur handoff complete nahi hua {citations}."
        ).strip()
    return (
        "Hello! I am ZentroMart's AI customer-support assistant. A human "
        "colleague should review this case. No ticket was created and no "
        f"handoff was completed in this chat {citations}."
    ).strip()


def _deterministic_return_window_answer(
    question: str,
    reply_language: str,
    evidence: list[RetrievalHit],
) -> str | None:
    if not _RETURN_WINDOW_PATTERN.search(question):
        return None

    product_rows = _matching_product_rows(question, evidence)
    if not product_rows:
        return None

    ret_2_hits = [
        hit for hit in evidence
        if hit.metadata.get("section_id") == "RET-2"
        and hit.metadata.get("status") == "current"
    ]
    if not ret_2_hits:
        return None

    policy_text = "\n".join(hit.text for hit in ret_2_hits)
    # Match category and window (e.g., "fashion: 15 days")
    # The policy usually looks like: "Category: Window" or in a list
    category = str(product_rows[0][1].get("return_category", "")).strip().lower()
    if not category:
        return None

    # Search for the category and the number of days in the policy text
    pattern = re.compile(rf"{re.escape(category)}[:\s\-]+(\d+)\s+days", re.IGNORECASE)
    match = pattern.search(policy_text)
    if not match:
        return None

    days = match.group(1)
    if reply_language == "hi":
        return (
            f"Hello! I am ZentroMart's AI customer-support assistant. "
            f"{category.capitalize()} items ke liye return window {days} din hai [RET-2] [{product_rows[0][0].evidence_id}]."
        )
    if reply_language == "hi-en":
        return (
            f"Namaste, main ZentroMart ka AI customer-support assistant hoon. "
            f"{category.capitalize()} items ke liye return window {days} days hai [RET-2] [{product_rows[0][0].evidence_id}]."
        )
    return (
        f"Hello! I am ZentroMart's AI customer-support assistant. "
        f"The return window for {category.capitalize()} items is {days} days [RET-2] [{product_rows[0][0].evidence_id}]."
    )


def _deterministic_serviceability_answer(
    question: str,
    reply_language: str,
    evidence: list[RetrievalHit],
) -> str | None:
    if not _SERVICEABILITY_PATTERN.search(question):
        return None

    pincode_rows = [
        (hit, hit.metadata["row"])
        for hit in evidence
        if hit.metadata.get("csv_table") == "pincodes"
        and isinstance(hit.metadata.get("row"), dict)
    ]
    if not pincode_rows:
        return None

    # We assume the retriever got the relevant pincode if it's in evidence
    hit, row = pincode_rows[0]
    serviceable = str(row.get("serviceable", "")).strip().lower() == "true"
    pincode = row.get("pincode", "the requested area")

    if reply_language == "hi":
        status = "serviceable hai" if serviceable else "serviceable nahi hai"
        return (
            f"Hello! I am ZentroMart's AI customer-support assistant. "
            f"Pincode {pincode} {status} [pincodes.csv:{hit.metadata.get('record_key')}]."
        )
    if reply_language == "hi-en":
        status = "serviceable hai" if serviceable else "serviceable nahi hai"
        return (
            f"Namaste, main ZentroMart ka AI customer-support assistant hoon. "
            f"Pincode {pincode} {status} [pincodes.csv:{hit.metadata.get('record_key')}]."
        )
    status = "serviceable" if serviceable else "not serviceable"
    return (
        f"Hello! I am ZentroMart's AI customer-support assistant. "
        f"Pincode {pincode} is {status} [pincodes.csv:{hit.metadata.get('record_key')}]."
    )


def _citation_is_retrieved(
    answer: str,
    evidence: list[RetrievalHit],
) -> bool:
    identities = set()
    for hit in evidence:
        for value in (
            hit.evidence_id,
            hit.metadata.get("evidence_key"),
            hit.metadata.get("section_id"),
            hit.metadata.get("record_key"),
        ):
            if value:
                normalized = str(value).replace("::", ":")
                identities.update(part for part in normalized.split(":") if part)
                identities.add(str(value))

    citations = [
        citation.strip()
        for group in _CITATION_PATTERN.findall(answer)
        for citation in re.split(r"\s*[,;]\s*", group)
        if citation.strip()
    ]
    return bool(citations) and all(
        citation in identities
        or citation.rsplit("::", maxsplit=1)[-1] in identities
        for citation in citations
    )


def _required_citations(
    question: str,
    evidence: list[RetrievalHit],
    answer: str = "",
) -> set[str]:
    required: set[str] = set()
    sections = _section_ids(evidence)
    if _RETURN_PATTERN.search(question):
        historical = any(
            hit.metadata.get("retrieval_role") == "historical_reference"
            for hit in evidence
        )
        if _CONDITION_PATTERN.search(question):
            required.add("RET-4")
        elif not historical and (
            _FESTIVE_SALE_PATTERN.search(question) and "RET-8" in sections
        ):
            required.add("RET-8")
        elif not historical and _RETURN_WINDOW_PATTERN.search(question):
            required.add("RET-2")
        if (
            _PLUS_PATTERN.search(question)
            and re.search(r"\b(?:pickup|collection)\s+fee\b|\bcharged\b", question, re.I)
        ):
            required.update(("RET-6", "PLS-2"))
    if _WARRANTY_PATTERN.search(question):
        explicit_assured = bool(re.search(r"\bassured\b", question, re.IGNORECASE))
        if not explicit_assured or re.search(
            r"\b(?:brand|manufacturer)\b",
            question,
            re.IGNORECASE,
        ):
            required.add("WAR-1")
        product_rows = (
            _matching_product_rows(question, evidence)
            if _has_explicit_product_reference(question, evidence)
            else []
        )
        assured_product = any(
            str(row.get("assured", "")).casefold() == "true"
            for _, row in product_rows
        )
        if explicit_assured or assured_product:
            required.add("WAR-2")
        for hit, _ in product_rows:
            required.add(f"products.csv:{hit.metadata.get('record_key')}")
    if _REFUND_PATTERN.search(question):
        if _COD_PATTERN.search(question):
            required.update(("REF-2", "REF-3"))
        elif (
            _REFUND_TIMELINE_PATTERN.search(question)
            or _REFUND_METHOD_PATTERN.search(question)
        ):
            required.add("REF-2")
    if _DAMAGE_PATTERN.search(question) and "DWM-3" not in sections:
        required.add("DWM-1")
        if (
            _DAMAGE_RESOLUTION_PATTERN.search(question)
            or _DAMAGE_RESOLUTION_PATTERN.search(answer)
        ):
            required.add("DWM-2")
        if _PICKUP_FEE_PATTERN.search(question) or _PICKUP_FEE_PATTERN.search(
            answer
        ):
            required.add("RET-6")
    return required


def _citations_cover_required_ids(
    answer: str,
    required_ids: set[str],
    evidence: list[RetrievalHit],
) -> bool:
    if not required_ids:
        return True
    citations = {
        citation.strip()
        for group in _CITATION_PATTERN.findall(answer)
        for citation in re.split(r"\s*[,;]\s*", group)
        if citation.strip()
    }
    covered = set()
    for hit in evidence:
        aliases = {
            str(value)
            for value in (
                hit.evidence_id,
                hit.metadata.get("evidence_key"),
                hit.metadata.get("section_id"),
                hit.metadata.get("record_key"),
            )
            if value
        }
        for citation in citations:
            if citation in aliases or any(
                citation.endswith(f"::{alias}::")
                or f"::{alias}::" in citation
                or citation.endswith(f"::{alias}")
                for alias in aliases
            ):
                section_id = hit.metadata.get("section_id")
                if section_id:
                    covered.add(str(section_id))
                if hit.metadata.get("source_type") == "csv":
                    covered.add(
                        f"{hit.metadata.get('csv_table')}.csv:"
                        f"{hit.metadata.get('record_key')}"
                    )
    return required_ids.issubset(covered)


def answer_with_evidence(
    question: str,
    reply_language: str,
    retriever: EvidenceRetriever,
    top_k: int = 8,
    *,
    verified_order_facts: Mapping[str, object] | None = None,
) -> RAGAnswer:
    """Generate a one-turn answer using only retrieved evidence and its sources."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    order_facts = dict(verified_order_facts or {})
    order_context = "\n".join(
        f"{key}: {value}" for key, value in sorted(order_facts.items())
    )
    analysis_question = question
    if order_context:
        analysis_question = (
            f"{question}\nVerified order facts (from the authenticated order "
            f"lookup):\n{order_context}"
        )
    evidence = retriever.search(
        analysis_question,
        top_k=top_k,
        candidate_pool=16,
        authority_adjustment=0.03,
        relevance_window=0.002,
        max_product_rows=2,
    )
    dated_refund = calculate_refund_window(analysis_question, evidence)
    if dated_refund is not None:
        return RAGAnswer(
            answer=format_refund_window(dated_refund, reply_language),
            evidence=evidence,
            grounding_status="deterministic_policy_calculation",
        )

    cod_refund_options = format_cod_refund_options(
        question,
        evidence,
        reply_language,
    )
    if cod_refund_options is not None:
        return RAGAnswer(
            answer=cod_refund_options,
            evidence=evidence,
            grounding_status="deterministic_policy_options",
        )

    # Deterministic Return Window (Category -> Policy link)
    return_window = _deterministic_return_window_answer(
        analysis_question,
        reply_language,
        evidence,
    )
    if return_window is not None:
        return RAGAnswer(
            answer=return_window,
            evidence=evidence,
            grounding_status="deterministic_return_window",
        )

    # Deterministic Serviceability (Pincode check)
    serviceability = _deterministic_serviceability_answer(
        analysis_question,
        reply_language,
        evidence,
    )
    if serviceability is not None:
        return RAGAnswer(
            answer=serviceability,
            evidence=evidence,
            grounding_status="deterministic_serviceability",
        )

    high_value_damage = _high_value_damage_product(analysis_question, evidence)
    if high_value_damage is not None and {
        "DWM-3",
        "ESC-1",
        "ESC-2",
    }.issubset(_section_ids(evidence)):
        product_hit, _ = high_value_damage
        return RAGAnswer(
            answer=_high_value_damage_answer(reply_language, product_hit),
            evidence=evidence,
            required_action="create_ticket_then_priority_handoff",
            action_status="not_executed",
            grounding_status="deterministic_policy_workflow",
            decision="escalate",
        )

    if high_value_damage is None and _is_damage_reporting_request(analysis_question):
        damage_reporting_answer = _damage_reporting_answer(
            question,
            reply_language,
            evidence,
        )
        if damage_reporting_answer is None:
            return RAGAnswer(
                answer=_safe_uncertainty(reply_language),
                evidence=evidence,
                grounding_status="insufficient_reporting_policy_evidence",
                decision="clarify",
            )
        return RAGAnswer(
            answer=damage_reporting_answer,
            evidence=evidence,
            grounding_status="deterministic_policy_response",
        )

    if _UNRECOGNIZED_ORDER_PATTERN.search(analysis_question) and {
        "ESC-1",
        "ESC-2",
    }.issubset(_section_ids(evidence)):
        return RAGAnswer(
            answer=_unexecuted_handoff_answer(reply_language, evidence),
            evidence=evidence,
            required_action="create_ticket_then_human_handoff",
            action_status="not_executed",
            grounding_status="deterministic_policy_workflow",
            decision="escalate",
        )

    if _missing_required_sections(analysis_question, evidence):
        return RAGAnswer(
            answer=_safe_uncertainty(reply_language),
            evidence=evidence,
            grounding_status="insufficient_required_evidence",
            decision="clarify",
        )

    authority_guidance = retriever.authority_guidance()
    evidence_context = "\n\n".join(
        (
            f"Evidence ID: {hit.evidence_id}\n"
            f"Evidence role: {hit.metadata.get('retrieval_role', 'retrieved evidence')}\n"
            f"Source: {hit.metadata.get('source_path')}\n"
            f"Document type: {hit.metadata.get('doc_type', hit.metadata.get('source_type'))}\n"
            f"Status: {hit.metadata.get('status')}\n"
            f"Version: {hit.metadata.get('version')}\n"
            f"Section ID: {hit.metadata.get('section_id')}\n"
            f"Evidence text:\n{hit.text}"
        )
        for hit in evidence
    )
    system_prompt = (
        "You are ZentroMart's AI customer-support assistant. Introduce yourself "
        "briefly, be polite, and reply in the requested language. Use only the "
        "provided evidence for ZentroMart-specific facts. Return "
        "exactly one JSON object and no surrounding prose. For an intentional "
        "abstention, return exactly {\"kind\":\"abstention\"}; do not include "
        "an answer field. For any response containing answer text, return "
        "{\"kind\":\"factual_answer\",\"answer\":\"...\"} and include "
        "retrieved citations for every policy claim. Never invent "
        "policies, dates, eligibility, delivery commitments, refund timelines, "
        "or compensation. If the retrieved evidence contains a direct fact or a clear policy rule (including product catalog data) that answers the question, provide the answer with citations. Only abstain if the evidence is truly missing, explicitly contradictory, or insufficient to form a complete answer. Do not "
        "ask multiple questions at once. Do not infer a condition or eligibility rule "
        "from a product category alone; use the policy section that states "
        "that condition. When a product's `return_category` is provided in the retrieved evidence, use that category to identify the applicable rule in the Returns policy. Keep payment-method rules separate and do not apply "
        "a cash-on-delivery FAQ to another payment method. When discussing "
        "a cash-on-delivery FAQ to another payment method. When discussing "
        "warranty, distinguish manufacturer warranty from additional "
        "ZentroMart Assured coverage and use both the applicable policy and "
        "product row. Cite only retrieved evidence IDs that directly support "
        "the adjacent claim, in square brackets. Never claim an action or "
        "handoff was completed unless a tool result confirms success. "
        "Treat superseded documents as non-authoritative unless the current "
        "successor explicitly scopes the case; the retriever excludes them by "
        "default. When a source is labeled historical_reference, use it only "
        "to explain the historical rule explicitly scoped by the current "
        "successor; the current successor remains governing. Apply the "
        "following supplied knowledge-base precedence rules "
        "when sources conflict:\n\n"
        f"{authority_guidance}"
    )
    user_prompt = (
        f"Requested reply language: {reply_language}\n"
        f"Customer question: {question}\n\n"
        + (
            f"Verified order facts (use only when relevant, and do not infer "
            f"additional order details):\n{order_context}\n\n"
            if order_context
            else ""
        )
        +
        "Retrieved evidence (may be empty):\n"
        f"{evidence_context or '[No evidence retrieved.]'}"
    )
    response: Any = chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    try:
        response_kind, answer = _parse_generated_response(
            response.message.content
        )
    except ValueError:
        return RAGAnswer(
            answer=_safe_uncertainty(reply_language),
            evidence=evidence,
            grounding_status="invalid_structured_response",
            decision="clarify",
        )
    if response_kind == "abstention":
        return RAGAnswer(
            answer=_safe_uncertainty(reply_language),
            evidence=evidence,
            grounding_status="model_abstention",
            decision="clarify",
        )
    if answer is None:
        raise AssertionError("Factual model responses must include answer text.")
    if _ACTION_CLAIM_PATTERN.search(answer):
        return RAGAnswer(
            answer=_unexecuted_handoff_answer(reply_language, evidence),
            evidence=evidence,
            required_action="create_ticket_then_human_handoff",
            action_status="not_executed",
            grounding_status="unexecuted_action_claim_rejected",
            decision="escalate",
        )
    if (
        not _citation_is_retrieved(answer, evidence)
        or not _citations_cover_required_ids(
            answer,
            _required_citations(question, evidence, answer),
            evidence,
        )
    ):
        return RAGAnswer(
            answer=_safe_uncertainty(reply_language),
            evidence=evidence,
            grounding_status="citation_validation_failed",
            decision="clarify",
        )
    if _REFUSAL_PATTERN.search(answer):
        decision = "refuse"
    elif "?" in answer or "？" in answer:
        decision = "clarify"
    else:
        decision = "answer"
    return RAGAnswer(answer, evidence, decision=decision)
