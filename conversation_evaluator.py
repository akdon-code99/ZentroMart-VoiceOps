"""Offline evaluator for the supplied multi-turn conversation practice set.

The default responder and tools are deterministic local fixtures. They never
call Ollama or write tickets; results measure the evaluation harness and these
fixtures, not live-agent answer quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from conversation_runner import (
    AgentReply,
    ConversationContext,
    ConversationMessage,
    CustomerInput,
    run_conversations,
)

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET = _ROOT / "practice" / "conversations.csv"
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "conversation_eval"
_EVALUATION_COLUMNS = {
    "conversation_id",
    "turn",
    "scenario",
    "expected_decision",
    "must_include",
    "must_not_include",
    "expected_tool_calls",
    "gold_evidence",
}
_TOOL_NAMES = ("lookup_order", "create_ticket")
_CALL_PATTERN = re.compile(r"\b(lookup_order|create_ticket)\s*\(", re.IGNORECASE)
_ORDER_ARGUMENT_PATTERN = re.compile(r"order_id\s*=\s*([0-9]+)")
_PROOF_MARKER_PATTERN = re.compile(
    r"(?:last\s*(?:four|4)|digits?|pincode|pin\s*code|"
    r"आख़िरी चार अंक|आखिरी चार अंक|पिनकोड|पिन\s*कोड)",
    re.IGNORECASE,
)
_DIGITS_PATTERN = re.compile(r"(?<!\d)(?:\d[\s-]*){3,6}(?!\d)")
@dataclass(frozen=True)
class EvaluationTurn:
    message: ConversationMessage
    scenario: str
    expected_decision: str
    must_include: str
    must_not_include: str
    expected_tool_calls: str
    gold_evidence: str


@dataclass(frozen=True)
class ToolCall:
    conversation_id: str
    turn: int
    tool_name: str
    safe_arguments: Mapping[str, object]
    created: bool | None = None
    verified: bool | None = None


Responder = Callable[[CustomerInput, ConversationContext], str | AgentReply]


class OfflineFixtureTools:
    """Read-only local tool fixture with no ticket-file or network side effects."""

    def __init__(self, data_dir: Path = _ROOT / "data") -> None:
        self.orders = self._read_index(data_dir / "orders.csv", "order_id")
        self.customers = self._read_index(data_dir / "customers.csv", "customer_id")
        self.items: dict[str, list[dict[str, str]]] = {}
        with (data_dir / "order_items.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            for item in csv.DictReader(handle):
                self.items.setdefault(item["order_id"], []).append(item)

    @staticmethod
    def _read_index(path: Path, key: str) -> dict[str, dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {row[key]: row for row in csv.DictReader(handle)}

    def lookup_order(
        self, order_id: str, verification_value: str | None = None
    ) -> dict[str, object]:
        normalized_id = "".join(character for character in str(order_id) if character.isdigit())
        if not verification_value:
            return {"verification_required": True}
        order = self.orders.get(normalized_id)
        if order is None:
            return {"verified": False, "verification_required": True}
        customer = self.customers.get(order["customer_id"])
        if customer is None:
            return {"verified": False, "verification_required": True}
        proof = "".join(character for character in str(verification_value) if character.isdigit())
        valid = (
            len(proof) == 4 and proof == customer["phone"][-4:]
        ) or (
            len(proof) == 6 and proof == order["delivery_pincode"]
        )
        if not valid:
            return {"verified": False, "verification_required": True}
        return {
            "found": True,
            "verified": True,
            **order,
            "items": self.items.get(normalized_id, []),
            "customer_first_name": customer["name"].split()[0],
            "is_plus": customer["is_plus"] == "True",
        }

    @staticmethod
    def create_ticket(
        summary: str,
        order_id: str | None = None,
        reason: str | None = None,
        intent: str | None = None,
    ) -> dict[str, object]:
        del summary, order_id, reason, intent
        return {"created": True, "ticket_id": "TKT-OFFLINE-FIXTURE"}


class DeterministicFixtureResponder:
    """Safe, intentionally limited responder for offline harness validation."""

    def __call__(
        self, customer: CustomerInput, context: ConversationContext
    ) -> AgentReply:
        text = customer.customer_message.casefold()
        recent = " ".join(entry.content for entry in context.history[-6:]).casefold()
        combined = f"{recent} {text}"
        if any(term in combined for term in ("speak to a person", "human", "colleague", "इंसान")):
            return AgentReply(
                "I can request a support ticket. No human transfer is available here.",
                decision="escalate",
                required_action="human handoff requested",
            )
        if any(term in text for term in ("approve", "approved", " मंज़ूर", "स्वीकृत")) and any(
            term in text for term in ("replacement", "refund", "compensation", "replace")
        ):
            return AgentReply(
                "I cannot approve that remedy. I can request support review.",
                decision="escalate",
                required_action="remedy approval requested",
            )
        if not context.order_id and any(
            term in combined for term in ("order", "ऑर्डर")
        ):
            return AgentReply(
                "I’m an AI support assistant. What is your order ID?",
                decision="clarify",
            )
        if any(term in text for term in ("force", "must approve", "बिना जांच")):
            return AgentReply(
                "I can’t approve a request that conflicts with policy.",
                decision="refuse",
            )
        return AgentReply(
            "I can help with that. Please share the one detail needed to check the request.",
            decision="clarify",
        )


def _split_facts(value: str) -> list[list[str]]:
    return [
        [alternative.strip() for alternative in fact.split("/") if alternative.strip()]
        for fact in value.split(";")
        if fact.strip()
    ]


def _canonical_dates(text: str) -> str:
    month_numbers = {
        "january": "01", "jan": "01", "जनवरी": "01",
        "february": "02", "feb": "02", "फ़रवरी": "02", "फरवरी": "02",
        "march": "03", "mar": "03", "मार्च": "03",
        "april": "04", "apr": "04", "अप्रैल": "04",
        "may": "05", "मई": "05",
        "june": "06", "jun": "06", "जून": "06",
        "july": "07", "jul": "07", "जुलाई": "07",
        "august": "08", "aug": "08", "अगस्त": "08",
        "september": "09", "sep": "09", "सितंबर": "09",
        "october": "10", "oct": "10", "अक्टूबर": "10",
        "november": "11", "nov": "11", "नवंबर": "11",
        "december": "12", "dec": "12", "दिसंबर": "12",
    }
    result = re.sub(
        r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)",
        lambda match: (
            f"{match.group(1)}-{int(match.group(2)):02d}-"
            f"{int(match.group(3)):02d}"
        ),
        text,
    )
    month_pattern = "|".join(
        sorted((re.escape(name) for name in month_numbers), key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?<!\w)(\d{{1,2}})\s+({month_pattern})[,]?\s+(\d{{4}})(?!\d)",
        re.IGNORECASE,
    )
    return pattern.sub(
        lambda match: (
            f"{match.group(3)}-{month_numbers[match.group(2).casefold()]}-"
            f"{int(match.group(1)):02d}"
        ),
        result,
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    text = _canonical_dates(text)
    text = re.sub(r"\b(four|४)\b", "4", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _contains_fact(response: str, fact: str) -> bool:
    normalized_response = f" {_normalize(response)} "
    normalized_fact = _normalize(fact)
    return bool(normalized_fact) and f" {normalized_fact} " in normalized_response


def check_required_facts(response: str, rubric: str) -> list[dict[str, object]]:
    return [
        {
            "fact": " / ".join(alternatives),
            "passed": any(_contains_fact(response, alternative) for alternative in alternatives),
        }
        for alternatives in _split_facts(rubric)
    ]


def check_forbidden_facts(response: str, rubric: str) -> list[dict[str, object]]:
    return [
        {
            "phrase": " / ".join(alternatives),
            "violated": any(_contains_fact(response, phrase) for phrase in alternatives),
        }
        for alternatives in _split_facts(rubric)
    ]


def _expected_calls(value: str) -> list[dict[str, object]]:
    calls = []
    for match in _CALL_PATTERN.finditer(value):
        call_end = value.find(")", match.end())
        fragment = value[match.end():call_end] if call_end >= 0 else ""
        order_match = _ORDER_ARGUMENT_PATTERN.search(fragment)
        safe_arguments: dict[str, object] = {}
        if order_match:
            safe_arguments["order_id"] = order_match.group(1)
        if "verification_value" in fragment:
            safe_arguments["verification_value"] = "<redacted>"
        calls.append(
            {
                "tool_name": match.group(1).lower(),
                "safe_arguments": safe_arguments,
            }
        )
    return calls


def _proof_digits(text: str) -> str | None:
    marker = _PROOF_MARKER_PATTERN.search(text)
    if marker is None:
        return None
    match = _DIGITS_PATTERN.search(text, marker.end())
    return re.sub(r"\D", "", match.group()) if match else None


def _redact_values(text: str, messages: Sequence[ConversationMessage]) -> str:
    for message in messages:
        proof = _proof_digits(message.customer_message)
        if proof:
            text = re.sub(re.escape(proof), "<redacted>", text, flags=re.IGNORECASE)
    return text


def load_evaluation_dataset(
    path: str | Path = _DEFAULT_DATASET,
) -> OrderedDict[str, list[EvaluationTurn]]:
    """Read runtime input and rubric side-by-side without mixing their fields."""
    grouped: OrderedDict[str, list[EvaluationTurn]] = OrderedDict()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _EVALUATION_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Evaluation CSV missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                message = ConversationMessage(
                    conversation_id=(row.get("conversation_id") or "").strip(),
                    turn=int(row.get("turn") or ""),
                    language=(row.get("language") or "").strip(),
                    customer_message=(row.get("customer_message") or "").strip(),
                )
            except ValueError as error:
                raise ValueError(f"Invalid conversation turn on CSV row {row_number}.") from error
            if not message.conversation_id or not message.language or not message.customer_message:
                raise ValueError(f"Missing runtime input on CSV row {row_number}.")
            grouped.setdefault(message.conversation_id, []).append(
                EvaluationTurn(
                    message=message,
                    scenario=(row.get("scenario") or "").strip(),
                    expected_decision=(row.get("expected_decision") or "").strip(),
                    must_include=(row.get("must_include") or "").strip(),
                    must_not_include=(row.get("must_not_include") or "").strip(),
                    expected_tool_calls=(row.get("expected_tool_calls") or "").strip(),
                    gold_evidence=(row.get("gold_evidence") or "").strip(),
                )
            )

    for conversation_id, turns in grouped.items():
        turns.sort(key=lambda turn: turn.message.turn)
        numbers = [turn.message.turn for turn in turns]
        if numbers != list(range(1, len(turns) + 1)):
            raise ValueError(
                f"Conversation {conversation_id} must have unique consecutive turns."
            )
    return grouped


def _safe_actual_arguments(
    tool_name: str, arguments: Mapping[str, object]
) -> dict[str, object]:
    safe = {
        key: value
        for key, value in arguments.items()
        if key in {"order_id", "reason", "intent"}
    }
    if tool_name == "lookup_order":
        safe["verification_value"] = (
            "<redacted>" if arguments.get("verification_value") else None
        )
    return safe


def _tool_calls_match(
    expected: Sequence[Mapping[str, object]],
    actual: Sequence[ToolCall],
) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_call, actual_call in zip(expected, actual):
        if expected_call["tool_name"] != actual_call.tool_name:
            return False
        expected_arguments = expected_call["safe_arguments"]
        for key, value in expected_arguments.items():
            actual_value = actual_call.safe_arguments.get(key)
            if key == "verification_value":
                actual_value = "<redacted>" if actual_value else None
            if actual_value != value:
                return False
    return True


def evaluate_dataset(
    dataset_path: str | Path = _DEFAULT_DATASET,
    *,
    responder: Responder | None = None,
    data_dir: str | Path = _ROOT / "data",
    lookup_tool: Callable[..., dict[str, object]] | None = None,
    ticket_creator: Callable[..., dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped = load_evaluation_dataset(dataset_path)
    rows: list[dict[str, object]] = []
    fixtures = OfflineFixtureTools(Path(data_dir))
    response_provider = responder or DeterministicFixtureResponder()

    for conversation_id, turns in grouped.items():
        messages = [turn.message for turn in turns]
        calls: list[ToolCall] = []
        private_values_by_turn: dict[int, set[str]] = {}
        turn_by_order = {
            re.sub(r"\D", "", turn.message.customer_message): turn.message.turn
            for turn in turns
            if "order" in turn.message.customer_message.casefold()
            or "ऑर्डर" in turn.message.customer_message
        }
        last_lookup_turn = 0
        active_responder_turn = 0

        def lookup_wrapper(
            order_id: str, verification_value: str | None = None
        ) -> dict[str, object]:
            nonlocal last_lookup_turn
            if verification_value is None:
                call_turn = turn_by_order.get(re.sub(r"\D", "", order_id), last_lookup_turn + 1)
            else:
                candidate = next(
                    (
                        turn.message.turn
                        for turn in turns
                        if turn.message.turn > last_lookup_turn
                        and _proof_digits(turn.message.customer_message)
                        == re.sub(r"\D", "", verification_value)
                    ),
                    last_lookup_turn + 1,
                )
                call_turn = candidate
            last_lookup_turn = call_turn
            result = (lookup_tool or fixtures.lookup_order)(
                order_id, verification_value
            )
            verified = result.get("found") is True and result.get("verified") is True
            if not verified:
                order = fixtures.orders.get(re.sub(r"\D", "", order_id), {})
                customer = fixtures.customers.get(order.get("customer_id", ""), {})
                protected = {
                    "customer_id": order.get("customer_id"),
                    "delivery_pincode": order.get("delivery_pincode"),
                    "status": order.get("status"),
                    "phone": customer.get("phone"),
                    "email": customer.get("email"),
                    "name": customer.get("name"),
                    "city": customer.get("city"),
                    "state": customer.get("state"),
                }
                private_values_by_turn.setdefault(call_turn, set()).update(
                    str(value)
                    for value in protected.values()
                    if value and len(str(value)) >= 4
                )
            calls.append(
                ToolCall(
                    conversation_id,
                    call_turn,
                    "lookup_order",
                    {
                        "order_id": order_id,
                        "verification_value": (
                            "<redacted>" if verification_value else None
                        ),
                    },
                    verified=verified,
                )
            )
            return result

        def ticket_wrapper(
            *,
            summary: str,
            order_id: str | None = None,
            reason: str | None = None,
            intent: str | None = None,
        ) -> dict[str, object]:
            result = (ticket_creator or fixtures.create_ticket)(
                summary=summary,
                order_id=order_id,
                reason=reason,
                intent=intent,
            )
            calls.append(
                ToolCall(
                    conversation_id,
                    active_responder_turn,
                    "create_ticket",
                    _safe_actual_arguments(
                        "create_ticket",
                        {
                            "order_id": order_id,
                            "reason": reason,
                            "intent": intent,
                        },
                    ),
                    created=result.get("created") is True,
                )
            )
            return result

        def observed_responder(
            customer: CustomerInput, context: ConversationContext
        ) -> str | AgentReply:
            nonlocal active_responder_turn
            active_responder_turn = len(context.history) // 2 + 1
            return response_provider(customer, context)

        run = run_conversations(
            messages,
            observed_responder,
            lookup_order_fn=lookup_wrapper,
            create_ticket_fn=ticket_wrapper,
        )
        result_by_turn = {result.turn: result for result in run.results}

        for turn in turns:
            result = result_by_turn[turn.message.turn]
            actual_calls = [call for call in calls if call.turn == turn.message.turn]
            expected_calls = _expected_calls(turn.expected_tool_calls)
            expected_names = [str(call["tool_name"]) for call in expected_calls]
            required_checks = check_required_facts(result.response, turn.must_include)
            forbidden_checks = check_forbidden_facts(result.response, turn.must_not_include)
            expected_evidence = [
                item.strip() for item in turn.gold_evidence.split(";") if item.strip()
            ]
            evidence_applicable = bool(expected_evidence)
            evidence_match = (
                bool(set(expected_evidence) & set(result.evidence_ids))
                if evidence_applicable
                else None
            )
            verification_leak = any(
                (proof := _proof_digits(message.customer_message))
                and proof in result.response
                for message in messages
            )
            order_private_leak = any(
                value.casefold() in result.response.casefold()
                for value in private_values_by_turn.get(turn.message.turn, set())
            )
            privacy_violation = order_private_leak or verification_leak
            ticket_expected = "create_ticket" in expected_names
            ticket_call_succeeded = all(
                call.created is True
                for call in actual_calls
                if call.tool_name == "create_ticket"
            ) if ticket_expected else None
            tool_match = _tool_calls_match(expected_calls, actual_calls)
            required_pass = all(bool(check["passed"]) for check in required_checks)
            forbidden_pass = not any(bool(check["violated"]) for check in forbidden_checks)
            decision_match = result.decision == turn.expected_decision
            passed = (
                decision_match
                and required_pass
                and forbidden_pass
                and tool_match
                and (evidence_match is not False)
                and not privacy_violation
                and (ticket_call_succeeded is not False)
            )
            reasons = []
            if not decision_match:
                reasons.append("decision_mismatch")
            if not required_pass:
                reasons.append("required_fact_missing")
            if not forbidden_pass:
                reasons.append("forbidden_fact_present")
            if not tool_match:
                reasons.append("tool_call_mismatch")
            if evidence_match is False:
                reasons.append("gold_evidence_not_present")
            if privacy_violation:
                reasons.append("privacy_violation")
            if ticket_call_succeeded is False:
                reasons.append("ticket_creation_failed")
            calls_json = [
                {
                    "tool_name": call.tool_name,
                    "arguments": dict(call.safe_arguments),
                    "created": call.created,
                    "verified": call.verified,
                }
                for call in actual_calls
            ]
            expected_calls_json = [
                {
                    "tool_name": call["tool_name"],
                    "arguments": dict(call["safe_arguments"]),
                }
                for call in expected_calls
            ]
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "turn": turn.message.turn,
                    "scenario": turn.scenario,
                    "expected_decision": turn.expected_decision,
                    "actual_decision": result.decision,
                    "response": _redact_values(result.response, messages),
                    "expected_tool_calls": json.dumps(expected_calls_json, ensure_ascii=False),
                    "actual_tool_calls": json.dumps(calls_json, ensure_ascii=False),
                    "tool_call_match": tool_match,
                    "required_fact_checks": json.dumps(required_checks, ensure_ascii=False),
                    "required_fact_count": len(required_checks),
                    "required_fact_pass_count": sum(bool(item["passed"]) for item in required_checks),
                    "forbidden_fact_checks": json.dumps(forbidden_checks, ensure_ascii=False),
                    "forbidden_fact_count": len(forbidden_checks),
                    "forbidden_violation_count": sum(bool(item["violated"]) for item in forbidden_checks),
                    "gold_evidence": "; ".join(expected_evidence),
                    "actual_evidence_ids": "; ".join(result.evidence_ids),
                    "evidence_applicable": evidence_applicable,
                    "evidence_match": evidence_match,
                    "privacy_violation": privacy_violation,
                    "ticket_created": ticket_call_succeeded,
                    "handoff_performed": result.handoff_performed,
                    "decision_match": decision_match,
                    "passed": passed,
                    "failure_reasons": ";".join(reasons),
                }
            )
    return rows, summarize_results(rows)


def summarize_results(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(rows)
    decisions = sum(bool(row["decision_match"]) for row in rows)
    required_total = sum(int(row["required_fact_count"]) for row in rows)
    required_passed = sum(int(row["required_fact_pass_count"]) for row in rows)
    forbidden_total = sum(int(row["forbidden_fact_count"]) for row in rows)
    forbidden_violations = sum(int(row["forbidden_violation_count"]) for row in rows)
    evidence_rows = [row for row in rows if row["evidence_applicable"]]
    evidence_matches = sum(row["evidence_match"] is True for row in evidence_rows)
    exact_tools = sum(bool(row["tool_call_match"]) for row in rows)
    expected_tool_count = 0
    actual_tool_count = 0
    true_positive_tools = 0
    for row in rows:
        expected = json.loads(str(row["expected_tool_calls"]))
        actual = json.loads(str(row["actual_tool_calls"]))
        expected_tool_count += len(expected)
        actual_tool_count += len(actual)
        expected_counts = Counter(item["tool_name"] for item in expected)
        actual_counts = Counter(item["tool_name"] for item in actual)
        true_positive_tools += sum(
            min(expected_counts[name], actual_counts[name]) for name in _TOOL_NAMES
        )
    precision = true_positive_tools / actual_tool_count if actual_tool_count else None
    recall = true_positive_tools / expected_tool_count if expected_tool_count else None
    scenario_decision_counts: dict[str, dict[str, int]] = {}
    expected_decision_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for group, key in (
            (scenario_decision_counts, str(row["scenario"])),
            (expected_decision_counts, str(row["expected_decision"])),
        ):
            bucket = group.setdefault(key, {"turns": 0, "passed": 0})
            bucket["turns"] += 1
            bucket["passed"] += int(bool(row["passed"]))

    return {
        "evaluation_mode": "offline_deterministic_fixture",
        "live_llm_or_tool_calls": False,
        "turn_count": total,
        "conversation_count": len({str(row["conversation_id"]) for row in rows}),
        "overall_pass_rate": sum(bool(row["passed"]) for row in rows) / total if total else None,
        "decision_accuracy": decisions / total if total else None,
        "required_fact_compliance": required_passed / required_total if required_total else None,
        "required_fact_denominator": required_total,
        "forbidden_fact_violation_rate": forbidden_violations / forbidden_total if forbidden_total else None,
        "forbidden_fact_denominator": forbidden_total,
        "evidence_support_rate": evidence_matches / len(evidence_rows) if evidence_rows else None,
        "evidence_applicable_turns": len(evidence_rows),
        "evidence_match_semantics": "Identifier overlap only; not a semantic support judgment.",
        "exact_tool_call_accuracy": exact_tools / total if total else None,
        "tool_call_precision": precision,
        "tool_call_recall": recall,
        "tool_precision_recall_basis": "Tool-name multiset counts; exact per-turn match also checks expected safe arguments.",
        "expected_tool_call_count": expected_tool_count,
        "actual_tool_call_count": actual_tool_count,
        "privacy_violation_count": sum(bool(row["privacy_violation"]) for row in rows),
        "scenario_breakdown": scenario_decision_counts,
        "expected_decision_breakdown": expected_decision_counts,
    }


def write_reports(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    csv_path = directory / f"conversation_turns_{run_id}.csv"
    json_path = directory / f"conversation_summary_{run_id}.json"
    markdown_path = directory / f"conversation_report_{run_id}.md"
    fields = list(rows[0]) if rows else []
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    failure_counts = Counter(
        reason
        for row in rows
        for reason in str(row["failure_reasons"]).split(";")
        if reason
    )
    report = [
        "# Offline conversation evaluation",
        "",
        "**Mode:** deterministic offline fixture; no LLM/API calls and no real tool side effects.",
        "**Interpretation:** metrics describe this fixture run and evaluator contract only, not live-agent quality.",
        "",
        f"- Turns: {summary['turn_count']} across {summary['conversation_count']} conversations.",
        f"- Overall fixture pass rate: {_percent(summary['overall_pass_rate'])}.",
        f"- Decision accuracy: {_percent(summary['decision_accuracy'])}.",
        f"- Required-fact compliance: {_percent(summary['required_fact_compliance'])} "
        f"({summary['required_fact_denominator']} fact checks).",
        f"- Forbidden-fact violation rate: {_percent(summary['forbidden_fact_violation_rate'])} "
        f"({summary['forbidden_fact_denominator']} checks; lower is better).",
        f"- Evidence identifier overlap: {_percent(summary['evidence_support_rate'])} "
        f"({summary['evidence_applicable_turns']} applicable turns); this is not semantic evidence validation.",
        f"- Exact per-turn tool-call accuracy: {_percent(summary['exact_tool_call_accuracy'])}; "
        f"tool-call precision/recall: {_percent(summary['tool_call_precision'])} / "
        f"{_percent(summary['tool_call_recall'])}.",
        f"- Privacy violations detected: {summary['privacy_violation_count']}.",
        "",
        "## Metric definitions",
        "",
        "- Overall pass rate denominator: all evaluated turns; every applicable check must pass.",
        "- Decision and exact tool-call accuracy denominators: all turns, including turns with no expected tool calls.",
        "- Required-fact compliance denominator: individual required fact groups; alternatives separated by `/` count as one group.",
        "- Forbidden-fact violation rate denominator: individual forbidden fact groups; a zero rate is desirable.",
        "- Evidence denominator: turns with at least one gold evidence ID; match means identifier overlap only. Empty gold evidence is not a failure.",
        "- Tool precision/recall use multiset counts of tool names. Exact per-turn matches also check expected order IDs and proof presence; proof values are redacted.",
        "- `not applicable` means no rubric item exists for that check; it is excluded from that metric, not counted as a failure.",
        "",
        "## Most common failure flags",
        "",
    ]
    report.extend(
        f"- `{reason}`: {count}" for reason, count in failure_counts.most_common()
    )
    if not failure_counts:
        report.append("- None.")
    report.extend(
        [
            "",
            "## Limitations",
            "",
            "- The responder is deliberately limited and deterministic; no real RAG answer generation occurred.",
            "- Evidence evaluation checks ID overlap, not whether the cited text semantically supports each claim.",
            "- Required/forbidden fact matching uses normalized phrase alternatives and date/digit normalization; nuanced paraphrases can need manual review.",
            "- The local lookup fixture reads synthetic CSV data to validate verification but does not call `tools.lookup_order`; ticket creation is a no-write fixture.",
            "- A ticket is counted as created only when the injected fixture returns `created=True`. No human-handoff action exists, so handoff remains false.",
            "",
            "Scenario and expected-decision breakdowns are available in the summary JSON.",
            "",
        ]
    )
    with markdown_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report))
    return csv_path, json_path, markdown_path


def _percent(value: object) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    rows, summary = evaluate_dataset(arguments.dataset)
    report_paths = write_reports(rows, summary, arguments.output_dir)
    for path in report_paths:
        print(path)
    print(
        f"Offline fixture evaluation: {summary['turn_count']} turns, "
        f"pass rate {_percent(summary['overall_pass_rate'])}."
    )


if __name__ == "__main__":
    main()
