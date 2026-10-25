"""Offline conversation-state and runner skeleton for the VoiceOps capstone."""

import csv
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONVERSATIONS_PATH = _PROJECT_ROOT / "practice" / "conversations.csv"
_REQUIRED_COLUMNS = {
    "conversation_id",
    "turn",
    "language",
    "customer_message",
}
_ORDER_ID_PATTERN = re.compile(
    r"(?:order(?:\s+(?:number|no\.?))?|ऑर्डर(?:\s+नंबर)?)"
    r"\D{0,30}((?:[0-9][\s-]*){7}[0-9])",
    re.IGNORECASE,
)
_VERIFICATION_PROOF_PATTERN = re.compile(
    r"(?P<pin>pincode|pin\s*code|पिनकोड|पिन\s*कोड)"
    r"|(?P<phone>last\s*(?:four|4)(?:\s+digits?)?|"
    r"aakhri(?:\s+(?:ke\s+)?(?:char|4))?|आख़िरी चार अंक|आखिरी चार अंक)",
    re.IGNORECASE,
)
_DIGIT_SEQUENCE_PATTERN = re.compile(r"(?<![0-9])(?:[0-9][\s-]*){3,5}[0-9](?![0-9])")
_ORDER_INDEPENDENT_PATTERN = re.compile(
    r"\b(?:coupon|coupons|promo(?:tion)?|discount|next order)\b|"
    r"कूपन|छूट",
    re.IGNORECASE,
)
_ORDER_SPECIFIC_PATTERN = re.compile(
    r"\b(?:my order|this order|that order|refund|delivery|shipment|shipped|"
    r"tracking|return|cancel|replacement|where is it|where's it)\b|"
    r"मेरा ऑर्डर|मेरे ऑर्डर|रिफंड|डिलीवरी|वापस|रद्द",
    re.IGNORECASE,
)
_MAX_VERIFICATION_ATTEMPTS = 2
_ORDER_SNAPSHOT_FIELDS = {
    "status",
    "payment_method",
    "delivery_type",
    "placed_at",
    "promised_date",
    "shipped_on",
    "delivered_on",
    "delivery_attempts",
    "subtotal_inr",
    "delivery_fee_inr",
    "cod_fee_inr",
    "total_inr",
    "cancellation_refund_initiated_on",
    "cancellation_refund_method",
    "cancellation_refund_amount_inr",
    "cancellation_rrn",
    "rto_refund_initiated_on",
    "rto_refund_method",
    "rto_refund_amount_inr",
    "rto_rrn",
    "express_late_credit_inr",
    "express_late_credit_on",
    "return_sku",
    "return_reason",
    "return_status",
    "return_line_no",
    "return_qty",
    "return_requested_on",
    "return_pickup_on",
    "return_qc_passed_on",
    "return_refund_initiated_on",
    "return_refund_method",
    "return_refund_amount_inr",
    "return_refund_completed_on",
    "return_rrn",
    "return_reverse_pickup_fee_inr",
    "damage_sku",
    "damage_issue",
    "damage_reported_at",
    "damage_hours_after_delivery",
    "damage_photos_attached",
    "damage_status",
    "contacts_last_7_days",
    "timeline",
    "is_plus",
}
_ORDER_ITEM_FIELDS = {
    "sku",
    "title",
    "return_category",
    "qty",
    "unit_price_inr",
}
_RETURN_ITEM_FIELDS = {"sku", "title", "return_category", "qty"}
_DAMAGE_ITEM_FIELDS = {"sku", "title", "qty", "unit_price_inr"}


@dataclass(frozen=True)
class ConversationMessage:
    """Customer-facing input only; evaluation columns are deliberately omitted."""

    conversation_id: str
    turn: int
    language: str
    customer_message: str


@dataclass(frozen=True)
class CustomerInput:
    """Only the current customer text and reply language passed to a responder."""

    language: str
    customer_message: str


@dataclass(frozen=True)
class HistoryEntry:
    role: str
    content: str


@dataclass(frozen=True)
class ConversationContext:
    """Safe runtime context made available to a future answer-generation layer."""

    history: tuple[HistoryEntry, ...]
    issue_context: tuple[str, ...]
    current_topic: str
    order_id: str | None
    verification_status: str
    verification_attempts: int
    ticket_created: bool
    verified_order_facts: Mapping[str, object]
    ticket_action_keys: tuple[str, ...]


@dataclass
class ConversationState:
    """Mutable state scoped to exactly one conversation."""

    conversation_id: str
    history: list[HistoryEntry] = field(default_factory=list)
    issue_context: list[str] = field(default_factory=list)
    current_topic: str = ""
    order_id: str | None = None
    verification_status: str = "unverified"
    verification_attempts: int = 0
    verification_attempts_by_order: dict[str, int] = field(default_factory=dict)
    lookup_requested_for_order: str | None = None
    verified_order_id: str | None = None
    verified_order_snapshot: dict[str, object] = field(default_factory=dict)
    ticket_created: bool = False
    ticket_action_results: dict[str, tuple[bool, str | None]] = field(
        default_factory=dict
    )

    def prepare_context(self, message: ConversationMessage) -> ConversationContext:
        previous_issue_context = tuple(self.issue_context)
        self.current_topic = message.customer_message
        self.issue_context.append(message.customer_message)
        order_id_match = _ORDER_ID_PATTERN.search(message.customer_message)
        if order_id_match:
            self.order_id = re.sub(r"[\s-]", "", order_id_match.group(1))

        attempts = self.verification_attempts_by_order.get(self.order_id or "", 0)
        self.verification_attempts = attempts
        if self.verified_order_id == self.order_id and self.order_id is not None:
            self.verification_status = "verified"

        facts_question = message.customer_message
        if message.customer_message == "[verification proof redacted]":
            facts_question = " ".join(previous_issue_context[-3:])
        relevant_order_facts = _order_facts_for_question(
            self.verified_order_snapshot,
            facts_question,
        )
        return ConversationContext(
            history=tuple(self.history),
            issue_context=tuple(self.issue_context),
            current_topic=self.current_topic,
            order_id=self.order_id,
            verification_status=self.verification_status,
            verification_attempts=self.verification_attempts,
            ticket_created=self.ticket_created,
            verified_order_facts=relevant_order_facts,
            ticket_action_keys=tuple(self.ticket_action_results),
        )

    def record_exchange(self, message: ConversationMessage, response: str) -> None:
        self.history.extend(
            (
                HistoryEntry(role="user", content=message.customer_message),
                HistoryEntry(role="assistant", content=response),
            )
        )


@dataclass(frozen=True)
class ConversationResult:
    conversation_id: str
    turn: int
    customer_message: str
    response: str
    decision: str = "answer"
    evidence_ids: tuple[str, ...] = ()
    ticket_created: bool = False
    handoff_performed: bool = False


@dataclass(frozen=True)
class ConversationRun:
    results: tuple[ConversationResult, ...]
    states: Mapping[str, ConversationState]


OrderLookup = Callable[..., dict]
TicketCreator = Callable[..., dict]


@dataclass(frozen=True)
class AgentReply:
    response: str
    decision: str = "answer"
    required_action: str = ""
    evidence_ids: tuple[str, ...] = ()
    grounding_status: str = ""


Responder = Callable[[CustomerInput, ConversationContext], str | AgentReply]


@dataclass(frozen=True)
class _VerificationProof:
    marked_as_proof: bool
    value: str | None


def _verification_proof(
    text: str,
    *,
    allow_bare_value: bool = False,
) -> _VerificationProof:
    marker = _VERIFICATION_PROOF_PATTERN.search(text)
    if marker is None:
        if allow_bare_value and re.fullmatch(r"[\s0-9-]+", text):
            digits = re.sub(r"\D", "", text)
            if len(digits) in (4, 6):
                return _VerificationProof(marked_as_proof=True, value=digits)
        return _VerificationProof(marked_as_proof=False, value=None)

    sequence = _DIGIT_SEQUENCE_PATTERN.search(text, marker.end())
    if sequence is None:
        return _VerificationProof(marked_as_proof=True, value=None)

    digits = re.sub(r"\D", "", sequence.group(0))
    expected_length = 6 if marker.lastgroup == "pin" else 4
    if len(digits) != expected_length:
        return _VerificationProof(marked_as_proof=True, value=None)
    return _VerificationProof(marked_as_proof=True, value=digits)


def _safe_customer_message(
    message: ConversationMessage,
    *,
    verification_pending: bool = False,
) -> ConversationMessage:
    proof = _verification_proof(
        message.customer_message,
        allow_bare_value=True,
    )
    possible_digits = re.search(r"(?<![0-9])(?:[0-9][\s-]*){3,5}[0-9](?![0-9])",
                                message.customer_message)
    if not proof.marked_as_proof and not (
        verification_pending and possible_digits is not None
    ):
        return message
    return ConversationMessage(
        conversation_id=message.conversation_id,
        turn=message.turn,
        language=message.language,
        customer_message="[verification proof redacted]",
    )


def _verification_prompt(language: str, attempts: int) -> str:
    if attempts >= _MAX_VERIFICATION_ATTEMPTS:
        if language == "hi":
            return (
                "मैं ऑर्डर की जानकारी साझा नहीं कर सकता क्योंकि सत्यापन "
                "नहीं हो पाया। सामान्य नीति में मदद कर सकता हूँ; कृपया "
                "रजिस्टर्ड फ़ोन से सपोर्ट से संपर्क करें।"
            )
        if language == "hi-en":
            return (
                "I couldn’t verify you, so main order details share nahi "
                "kar sakta. General policy mein help kar sakta hoon; please "
                "registered phone se support contact karein."
            )
        return (
            "I couldn’t verify the caller, so I can’t share order details. "
            "I can still help with general policy questions; please contact "
            "support from the registered phone number."
        )
    if attempts == 1:
        if language == "hi":
            return (
                "सत्यापन नहीं मिला। कृपया एक बार और रजिस्टर्ड फ़ोन के आख़िरी "
                "चार अंक या डिलीवरी पिनकोड दें। सत्यापन तक मैं ऑर्डर की "
                "जानकारी साझा नहीं कर सकता।"
            )
        if language == "hi-en":
            return (
                "Verification match nahi hua. Ek baar aur registered phone "
                "ke last four digits ya delivery pincode try karein. "
                "Verification ke bina main order details share nahi kar sakta."
            )
        return (
            "That verification did not match. Please try once more with "
            "either the registered phone’s last four digits or the delivery "
            "pincode. I can’t share order details until verification succeeds."
        )
    if language == "hi":
        return (
            "कृपया रजिस्टर्ड फ़ोन के आख़िरी चार अंक या इस ऑर्डर का डिलीवरी "
            "पिनकोड बताएं। सत्यापन तक मैं ऑर्डर की जानकारी साझा नहीं कर सकता।"
        )
    if language == "hi-en":
        return (
            "Please registered phone ke last four digits ya is order ka "
            "delivery pincode batayein. Verification ke bina main order "
            "details share nahi kar sakta."
        )
    return (
        "Please share either the registered phone’s last four digits or the "
        "delivery pincode for this order. I can’t share order details until "
        "verification is complete."
    )


def _is_order_independent_question(text: str) -> bool:
    return (
        _ORDER_INDEPENDENT_PATTERN.search(text) is not None
        and _ORDER_SPECIFIC_PATTERN.search(text) is None
    )


def _call_lookup(
    lookup_order_fn: OrderLookup | None,
    order_id: str,
    verification_value: str | None = None,
) -> dict | None:
    if lookup_order_fn is None:
        return None
    if verification_value is None:
        return lookup_order_fn(order_id)
    return lookup_order_fn(order_id, verification_value=verification_value)


def _verified_lookup_result(result: object) -> bool:
    return (
        isinstance(result, dict)
        and result.get("found") is True
        and result.get("verified") is True
    )


def _verification_required_result(result: object) -> bool:
    return (
        isinstance(result, dict)
        and result.get("verification_required") is True
        and result.get("verified") is not True
    )


def _sanitize_order_result(result: Mapping[str, object]) -> dict[str, object]:
    """Retain only operational order fields; never cache the raw tool result."""
    snapshot = {
        key: result[key]
        for key in _ORDER_SNAPSHOT_FIELDS
        if key in result and result[key] not in (None, "")
    }
    raw_items = result.get("items")
    if isinstance(raw_items, list):
        items = []
        for item in raw_items:
            if isinstance(item, Mapping):
                safe_item = {
                    key: item[key]
                    for key in _ORDER_ITEM_FIELDS
                    if key in item and item[key] not in (None, "")
                }
                if safe_item:
                    items.append(safe_item)
        if items:
            snapshot["items"] = items
    return snapshot


def _order_facts_for_question(
    snapshot: Mapping[str, object],
    question: str,
) -> dict[str, object]:
    folded = question.casefold()
    facts: dict[str, object] = {}

    if _is_order_independent_question(question):
        return facts

    if re.search(r"\b(where|status|track|ship|shipped|deliver|delivery|home)\b|"
                 r"कहाँ|स्थिति|डिलीवरी|पहुंच", folded):
        for key in (
            "status", "delivery_type", "promised_date", "shipped_on",
            "delivered_on", "delivery_attempts", "timeline",
        ):
            if key in snapshot:
                facts[key] = snapshot[key]

    if re.search(r"\b(refund|money back|rrn|reference number)\b|रिफंड|पैसे", folded):
        refund_groups = {
            "return": (
                "return_refund_initiated_on",
                "return_refund_method",
                "return_refund_amount_inr",
                "return_refund_completed_on",
                "return_rrn",
                "return_status",
            ),
            "cancellation": (
                "cancellation_refund_initiated_on",
                "cancellation_refund_method",
                "cancellation_refund_amount_inr",
                "cancellation_rrn",
            ),
            "rto": (
                "rto_refund_initiated_on",
                "rto_refund_method",
                "rto_refund_amount_inr",
                "rto_rrn",
            ),
        }
        if re.search(r"\b(return|returned)\b|रिटर्न|वापस", folded):
            selected_group = "return"
        elif re.search(r"\b(cancel|cancellation)\b|रद्द", folded):
            selected_group = "cancellation"
        else:
            populated_groups = [
                name
                for name, keys in refund_groups.items()
                if any(snapshot.get(key) is not None for key in keys)
            ]
            selected_group = populated_groups[0] if len(populated_groups) == 1 else None
        selected_keys = (
            refund_groups[selected_group]
            if selected_group is not None
            else ()
        )
        keys = ("payment_method", *selected_keys)
        if re.search(r"\b(rrn|reference number)\b", folded):
            keys = (*keys, "cancellation_rrn", "rto_rrn", "return_rrn")
        for key in keys:
            if key in snapshot:
                facts[key] = snapshot[key]

    if re.search(r"\b(return|eligible|eligibility|warranty|assured)\b|"
                 r"वापस|रिटर्न|वारंटी|गारंटी", folded):
        for key in (
            "placed_at", "delivered_on", "return_sku", "return_reason",
            "return_status", "return_requested_on", "return_pickup_on",
            "return_qc_passed_on", "is_plus",
        ):
            if key in snapshot:
                facts[key] = snapshot[key]
        relevant_items = _relevant_order_items(
            snapshot,
            question,
            _RETURN_ITEM_FIELDS,
        )
        if relevant_items:
            facts["items"] = relevant_items

    if re.search(r"\b(damag|broken|defect|wrong item|missing item|crack|"
                 r"high.value)\b|खराब|टूटा|फटा|क्षतिग्रस्त", folded):
        for key in (
            "damage_sku", "damage_issue", "damage_reported_at",
            "damage_hours_after_delivery", "damage_photos_attached",
            "damage_status",
        ):
            if key in snapshot:
                facts[key] = snapshot[key]
        relevant_items = _relevant_order_items(
            snapshot,
            question,
            _DAMAGE_ITEM_FIELDS,
        )
        if relevant_items:
            facts["items"] = relevant_items

    if re.search(r"\b(cancel|cancellation|payment|paid|amount|charge|cost)\b|"
                 r"रद्द|भुगतान|राशि|पैसे", folded):
        for key in (
            "status", "payment_method", "placed_at", "shipped_on",
            "total_inr", "subtotal_inr", "delivery_fee_inr", "cod_fee_inr",
            "cancellation_refund_initiated_on", "cancellation_refund_method",
            "cancellation_refund_amount_inr",
        ):
            if key in snapshot:
                facts[key] = snapshot[key]

    if re.search(r"\b(contact|called|again|repeat)\b|संपर्क|फिर", folded):
        if "contacts_last_7_days" in snapshot:
            facts["contacts_last_7_days"] = snapshot["contacts_last_7_days"]

    if not facts and _ORDER_SPECIFIC_PATTERN.search(question):
        for key in ("status", "payment_method", "items"):
            if key in snapshot:
                facts[key] = snapshot[key]
    return facts


def _relevant_order_items(
    snapshot: Mapping[str, object],
    question: str,
    allowed_fields: set[str],
) -> list[dict[str, object]]:
    items = snapshot.get("items")
    if not isinstance(items, list):
        return []
    folded_question = question.casefold()
    matching_items = [
        item
        for item in items
        if isinstance(item, Mapping)
        and (
            str(item.get("title", "")).casefold() in folded_question
            or str(item.get("sku", "")).casefold() in folded_question
        )
    ]
    if not matching_items and len(items) == 1:
        matching_items = [items[0]] if isinstance(items[0], Mapping) else []
    return [
        {
            key: item[key]
            for key in allowed_fields
            if key in item and item[key] not in (None, "")
        }
        for item in matching_items
    ]


def _verification_response(
    message: ConversationMessage,
    state: ConversationState,
    lookup_order_fn: OrderLookup | None,
) -> str | None:
    prior_order_id = state.order_id
    order_id_match = _ORDER_ID_PATTERN.search(message.customer_message)
    if order_id_match:
        state.order_id = re.sub(r"[\s-]", "", order_id_match.group(1))
    order_id = state.order_id
    proof = _verification_proof(
        message.customer_message,
        allow_bare_value=(
            order_id is not None and state.verification_status == "pending"
        ),
    )
    new_order_id = order_id is not None and order_id != prior_order_id
    explicit_order_id = order_id_match is not None

    order_workflow_required = (
        not _is_order_independent_question(message.customer_message)
        and (
            explicit_order_id
            or (
                order_id is not None
                and state.verified_order_id != order_id
            )
        )
    )
    if not order_workflow_required:
        return None

    attempts = state.verification_attempts_by_order.get(order_id or "", 0)
    state.verification_attempts = attempts
    if state.verified_order_id == order_id and order_id is not None:
        return None
    if attempts >= _MAX_VERIFICATION_ATTEMPTS:
        state.verification_status = "exhausted"
        return _verification_prompt(message.language, attempts)

    if proof.value is not None:
        if order_id is None:
            return None
        if lookup_order_fn is None:
            state.verification_status = "unavailable"
            return (
                "I can’t access order verification right now, so I can’t share "
                "order details. Please contact support or ask a general policy question."
            )
        state.verification_attempts_by_order[order_id] = attempts + 1
        state.verification_attempts = attempts + 1
        result = _call_lookup(lookup_order_fn, order_id, proof.value)
        if _verified_lookup_result(result):
            state.verification_status = "verified"
            state.verified_order_id = order_id
            state.lookup_requested_for_order = order_id
            state.verified_order_snapshot = _sanitize_order_result(result)
            return None
        state.verification_status = (
            "exhausted"
            if state.verification_attempts >= _MAX_VERIFICATION_ATTEMPTS
            else "pending"
        )
        return _verification_prompt(message.language, state.verification_attempts)

    if (
        order_id is not None
        and (new_order_id or state.lookup_requested_for_order != order_id)
    ):
        state.lookup_requested_for_order = order_id
        if lookup_order_fn is None:
            state.verification_status = "pending"
        else:
            result = _call_lookup(lookup_order_fn, order_id)
            if _verification_required_result(result):
                state.verification_status = "pending"
            else:
                state.verification_status = "unavailable"
                return (
                    "I couldn’t start order verification, so I can’t share order "
                    "details. Please contact support or ask a general policy question."
                )

    return _verification_prompt(message.language, attempts)


def _ticket_action_key(
    context: ConversationContext,
    required_action: str,
) -> str:
    issue_text = " ".join(context.issue_context).casefold()
    intent_text = context.current_topic.casefold() or issue_text
    if re.search(r"damag|broken|defect|wrong item|crack|ખરાબ|टૂटा|टूटा", intent_text):
        intent = "damaged_item"
    elif re.search(r"unauthori[sz]ed|not my order|did not place", intent_text):
        intent = "unauthorized_activity"
    elif re.search(r"refund|रिफंड", intent_text):
        intent = "refund_issue"
    elif re.search(r"\bhuman\b|\bagent\b|representative", intent_text):
        intent = "human_requested"
    elif re.search(r"battery|burning|smoke|spark|injury|safety", intent_text):
        intent = "safety_risk"
    elif re.search(r"legal|court|police|कानूनी|पुलिस", intent_text):
        intent = "legal_escalation"
    else:
        words = re.findall(r"[a-z0-9]+", intent_text)
        intent = "_".join(words[-8:]) or "unspecified"
        if re.search(r"damag|broken|defect|wrong item|crack", issue_text):
            intent = "damaged_item"
        elif re.search(r"unauthori[sz]ed|not my order|did not place", issue_text):
            intent = "unauthorized_activity"
        elif re.search(r"refund", issue_text):
            intent = "refund_issue"
        elif re.search(r"\bhuman\b|\bagent\b|representative", issue_text):
            intent = "human_requested"
        elif re.search(r"battery|burning|smoke|spark|injury|safety", issue_text):
            intent = "safety_risk"
        elif re.search(r"legal|court|police", issue_text):
            intent = "legal_escalation"
    return f"{context.order_id or 'no-order'}|{required_action}|{intent}"


def _ticket_summary(
    context: ConversationContext,
    required_action: str,
) -> str:
    issue = " | ".join(context.issue_context[-6:])
    return (
        f"Issue: {issue}. Order ID: {context.order_id or 'not provided'}. "
        f"Intent: {required_action}. Checked: caller verification status is "
        f"{context.verification_status}. Escalation reason: {required_action}."
    )


def _ticket_response(
    ticket_result: Mapping[str, object],
    previous_result: tuple[bool, str | None] | None,
    evidence_ids: tuple[str, ...],
    required_action: str,
) -> tuple[str, bool]:
    citation = " [ESC-2]" if "ESC-2" in evidence_ids else ""
    review_target = (
        " This case requires review by Priority Resolution."
        if "priority" in required_action.casefold()
        else ""
    )
    if previous_result is not None:
        created, ticket_id = previous_result
        if created and ticket_id:
            return (
                f"A support ticket was already created (ID: {ticket_id})"
                f"{citation}.{review_target} No direct human handoff is available here, so "
                "I have not transferred you."
            ), True
        return (
            "A ticket was already attempted for this issue but was not "
            "created successfully. No human handoff has occurred."
        ), False

    if ticket_result.get("created") is True:
        ticket_id = str(ticket_result.get("ticket_id", ""))
        if ticket_id:
            return (
                f"I created support ticket {ticket_id}{citation}.{review_target} No direct "
                "human handoff is available here, so I have not transferred you."
            ), True
        return (
            "The ticket tool reported success but did not return a ticket ID. "
            "I cannot confirm the ticket details, and no human handoff occurred."
        ), True
    return (
        "I could not create the required support ticket, so no human handoff "
        "has occurred. Please contact support directly."
    ), False


def load_conversations(
    path: str | Path = _CONVERSATIONS_PATH,
) -> list[ConversationMessage]:
    """Load only runtime message fields, never evaluation labels or notes."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Conversation CSV is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

        messages = []
        for row_number, row in enumerate(reader, start=2):
            conversation_id = (row.get("conversation_id") or "").strip()
            customer_message = (row.get("customer_message") or "").strip()
            language = (row.get("language") or "").strip()
            try:
                turn = int(row.get("turn") or "")
            except ValueError as error:
                raise ValueError(
                    f"Invalid turn number on CSV row {row_number}."
                ) from error
            if not conversation_id or not customer_message or not language:
                raise ValueError(
                    f"Conversation ID, language, and customer message are "
                    f"required on CSV row {row_number}."
                )
            if turn < 1:
                raise ValueError(f"Turn numbers must be positive (row {row_number}).")
            messages.append(
                ConversationMessage(
                    conversation_id=conversation_id,
                    turn=turn,
                    language=language,
                    customer_message=customer_message,
                )
            )

    return _group_and_order(messages)


def _group_and_order(
    messages: Iterable[ConversationMessage],
) -> list[ConversationMessage]:
    grouped: OrderedDict[str, list[ConversationMessage]] = OrderedDict()
    for message in messages:
        grouped.setdefault(message.conversation_id, []).append(message)

    ordered = []
    for conversation_id, turns in grouped.items():
        turns.sort(key=lambda message: message.turn)
        turn_numbers = [message.turn for message in turns]
        expected_turns = list(range(1, len(turns) + 1))
        if turn_numbers != expected_turns:
            raise ValueError(
                f"Conversation {conversation_id} must have unique, consecutive "
                "turns starting at 1."
            )
        ordered.extend(turns)
    return ordered


class ConversationRunner:
    """Run turns incrementally while retaining each conversation's state."""

    def __init__(
        self,
        responder: Callable[[CustomerInput, ConversationContext], str | AgentReply],
        lookup_order_fn: OrderLookup | None = None,
        create_ticket_fn: TicketCreator | None = None,
    ) -> None:
        self._responder = responder
        self._lookup_order_fn = lookup_order_fn
        self._create_ticket_fn = create_ticket_fn
        self.states: dict[str, ConversationState] = {}
        self._results: list[ConversationResult] = []
        self._next_turn: dict[str, int] = {}

    def run_turn(self, message: ConversationMessage) -> ConversationResult:
        """Process the next turn once; earlier turns and tool calls are not replayed."""
        expected_turn = self._next_turn.get(message.conversation_id, 1)
        if message.turn != expected_turn:
            raise ValueError(
                f"Conversation {message.conversation_id} expected turn "
                f"{expected_turn}, received {message.turn}."
            )

        state = self.states.setdefault(
            message.conversation_id,
            ConversationState(conversation_id=message.conversation_id),
        )
        verification_response = _verification_response(
            message,
            state,
            self._lookup_order_fn,
        )
        safe_message = _safe_customer_message(
            message,
            verification_pending=state.verification_status in {"pending", "exhausted"},
        )
        context = state.prepare_context(safe_message)
        if verification_response is None:
            customer_input = CustomerInput(
                language=safe_message.language,
                customer_message=safe_message.customer_message,
            )
            response = self._responder(customer_input, context)
        else:
            response = verification_response
        decision = (
            "refuse"
            if (
                verification_response is not None
                and state.verification_status == "exhausted"
            )
            else "clarify" if verification_response is not None
            else "answer"
        )
        evidence_ids: tuple[str, ...] = ()
        handoff_performed = False
        if isinstance(response, AgentReply):
            decision = response.decision
            evidence_ids = response.evidence_ids
            if response.required_action:
                decision = "escalate"
                action_key = _ticket_action_key(context, response.required_action)
                previous_result = state.ticket_action_results.get(action_key)
                if previous_result is not None:
                    response_text, _ = _ticket_response(
                        {},
                        previous_result,
                        evidence_ids,
                        response.required_action,
                    )
                elif self._create_ticket_fn is None:
                    response_text = (
                        "This issue requires a support ticket before human "
                        "escalation, but ticket creation is unavailable here. "
                        "No ticket or human handoff has occurred."
                    )
                else:
                    ticket_result = self._create_ticket_fn(
                        summary=_ticket_summary(context, response.required_action),
                        order_id=context.order_id,
                        reason=response.required_action,
                        intent=action_key.rsplit("|", maxsplit=1)[-1],
                    )
                    ticket_created = ticket_result.get("created") is True
                    ticket_id = (
                        str(ticket_result.get("ticket_id"))
                        if ticket_created and ticket_result.get("ticket_id")
                        else None
                    )
                    state.ticket_action_results[action_key] = (
                        ticket_created,
                        ticket_id,
                    )
                    state.ticket_created = state.ticket_created or ticket_created
                    response_text, _ = _ticket_response(
                        ticket_result,
                        None,
                        evidence_ids,
                        response.required_action,
                    )
                response = response_text
            else:
                response = response.response
        if not isinstance(response, str):
            raise TypeError("The conversation responder must return a string.")

        state.record_exchange(safe_message, response)
        result = ConversationResult(
            conversation_id=safe_message.conversation_id,
            turn=safe_message.turn,
            customer_message=safe_message.customer_message,
            response=response,
            decision=decision,
            evidence_ids=evidence_ids,
            ticket_created=state.ticket_created,
            handoff_performed=handoff_performed,
        )
        self._results.append(result)
        self._next_turn[message.conversation_id] = expected_turn + 1
        return result

    def snapshot(self) -> ConversationRun:
        """Return the results and current state without replaying any turn."""
        return ConversationRun(
            results=tuple(self._results),
            states=dict(self.states),
        )


def run_conversations(
    messages: Iterable[ConversationMessage],
    responder: Callable[[CustomerInput, ConversationContext], str | AgentReply],
    lookup_order_fn: OrderLookup | None = None,
    create_ticket_fn: TicketCreator | None = None,
) -> ConversationRun:
    """Run isolated conversations with an injected responder and optional lookup tool."""
    runner = ConversationRunner(
        responder,
        lookup_order_fn=lookup_order_fn,
        create_ticket_fn=create_ticket_fn,
    )
    for message in _group_and_order(messages):
        runner.run_turn(message)
    return runner.snapshot()
