import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from conversation_evaluator import (
    check_forbidden_facts,
    check_required_facts,
    evaluate_dataset,
    load_evaluation_dataset,
)
from conversation_runner import AgentReply


CSV_FIELDS = [
    "conversation_id",
    "turn",
    "scenario",
    "language",
    "order_id",
    "customer_id",
    "customer_message",
    "expected_decision",
    "must_include",
    "must_not_include",
    "expected_tool_calls",
    "gold_evidence",
    "notes",
]


def row(
    turn,
    message,
    *,
    conversation_id="PC001",
    expected_decision="answer",
    must_include="",
    must_not_include="",
    expected_tool_calls="",
    gold_evidence="",
    order_id="",
    customer_id="",
    scenario="test",
):
    return {
        "conversation_id": conversation_id,
        "turn": str(turn),
        "scenario": scenario,
        "language": "en",
        "order_id": order_id,
        "customer_id": customer_id,
        "customer_message": message,
        "expected_decision": expected_decision,
        "must_include": must_include,
        "must_not_include": must_not_include,
        "expected_tool_calls": expected_tool_calls,
        "gold_evidence": gold_evidence,
        "notes": "evaluation-only note",
    }


class ConversationEvaluatorTests(unittest.TestCase):
    def write_dataset(self, directory, rows):
        path = Path(directory) / "conversations.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_tool_data(self, directory):
        data = Path(directory) / "data"
        data.mkdir()
        with (data / "orders.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["order_id", "customer_id", "delivery_pincode", "status"])
            writer.writeheader()
            writer.writerow(
                {
                    "order_id": "12345678",
                    "customer_id": "C-PRIVATE",
                    "delivery_pincode": "560001",
                    "status": "Shipped",
                }
            )
        with (data / "customers.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["customer_id", "phone", "name", "is_plus"])
            writer.writeheader()
            writer.writerow(
                {
                    "customer_id": "C-PRIVATE",
                    "phone": "9999994533",
                    "name": "Private Customer",
                    "is_plus": "False",
                }
            )
        with (data / "order_items.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["order_id", "sku", "title", "qty"])
            writer.writeheader()
        return data

    def test_dataset_is_grouped_in_numeric_turn_order_and_keeps_rubric_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(2, "second", expected_decision="clarify"),
                    row(1, "first", order_id="99999999", customer_id="C-HIDDEN"),
                ],
            )
            grouped = load_evaluation_dataset(path)

        self.assertEqual([item.message.turn for item in grouped["PC001"]], [1, 2])
        self.assertEqual(grouped["PC001"][0].message.customer_message, "first")
        self.assertEqual(
            set(grouped["PC001"][0].message.__dataclass_fields__),
            {"conversation_id", "turn", "language", "customer_message"},
        )
        self.assertEqual(grouped["PC001"][0].scenario, "test")

    def test_required_alternatives_dates_and_forbidden_claims(self):
        response = "The deadline is 27 October 2026. Your last four digits are needed."
        required = check_required_facts(
            response,
            "2026-10-27 / 27 October 2026 ; last 4 / last four",
        )
        forbidden = check_forbidden_facts(
            "Your replacement has been approved.",
            "has been approved ; approve kar diya",
        )

        self.assertEqual([check["passed"] for check in required], [True, True])
        self.assertTrue(forbidden[0]["violated"])
        self.assertFalse(forbidden[1]["violated"])

    def test_decision_tool_and_evidence_are_compared_with_applicability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(
                        1,
                        "What is my deadline?",
                        must_include="2026-10-27 / 27 October 2026",
                        expected_decision="answer",
                        gold_evidence="RET-2",
                    ),
                    row(2, "Thanks.", expected_decision="answer"),
                ],
            )

            def responder(_customer, context):
                if context.current_topic == "What is my deadline?":
                    return AgentReply(
                        "The deadline is 27 October 2026.",
                        evidence_ids=("RET-2",),
                    )
                return AgentReply("You’re welcome.")

            results, summary = evaluate_dataset(path, responder=responder)

        self.assertTrue(results[0]["passed"])
        self.assertTrue(results[0]["evidence_match"])
        self.assertIsNone(results[1]["evidence_match"])
        self.assertEqual(summary["evidence_applicable_turns"], 1)
        self.assertEqual(summary["decision_accuracy"], 1.0)

    def test_tool_calls_are_logged_with_safe_arguments_and_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(
                        1,
                        "The order number is 12345678.",
                        expected_decision="clarify",
                        expected_tool_calls="lookup_order(order_id=12345678)",
                    ),
                    row(
                        2,
                        "Last four digits are 4533.",
                        expected_decision="clarify",
                        expected_tool_calls=(
                            "lookup_order(order_id=12345678, "
                            "verification_value=4533)"
                        ),
                    ),
                ],
            )
            results, summary = evaluate_dataset(
                path,
                data_dir=self.write_tool_data(directory),
            )

        actual_initial = json.loads(results[0]["actual_tool_calls"])
        actual_verified = json.loads(results[1]["actual_tool_calls"])
        self.assertEqual(actual_initial[0]["tool_name"], "lookup_order")
        self.assertEqual(actual_initial[0]["arguments"]["order_id"], "12345678")
        self.assertIsNone(actual_initial[0]["arguments"]["verification_value"])
        self.assertTrue(results[0]["tool_call_match"])
        self.assertTrue(actual_verified[0]["verified"])
        self.assertEqual(actual_verified[0]["arguments"]["verification_value"], "<redacted>")
        self.assertTrue(results[1]["tool_call_match"])
        self.assertNotIn("4533", json.dumps(results))
        self.assertEqual(summary["actual_tool_call_count"], 2)

    def test_tool_call_match_checks_expected_safe_order_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(
                        1,
                        "The order number is 12345678.",
                        expected_decision="clarify",
                        expected_tool_calls="lookup_order(order_id=87654321)",
                    )
                ],
            )
            results, _ = evaluate_dataset(path, data_dir=self.write_tool_data(directory))

        self.assertFalse(results[0]["tool_call_match"])
        self.assertIn("tool_call_mismatch", results[0]["failure_reasons"])

    def test_evaluation_only_fields_never_reach_responder(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(
                        1,
                        "Hello",
                        expected_decision="answer",
                        must_include="HIDDEN-FACT",
                        must_not_include="HIDDEN-BAD",
                        expected_tool_calls="create_ticket()",
                        gold_evidence="GOLD-SECTION",
                        order_id="99999999",
                        customer_id="C-HIDDEN",
                    )
                ],
            )

            def responder(customer, context):
                seen.append((customer, context))
                return AgentReply("Hello.")

            evaluate_dataset(path, responder=responder, data_dir=self.write_tool_data(directory))

        customer, context = seen[0]
        exposed = repr((customer, context))
        for hidden in ("HIDDEN-FACT", "HIDDEN-BAD", "GOLD-SECTION", "C-HIDDEN", "99999999", "evaluation-only"):
            self.assertNotIn(hidden, exposed)

    def test_failed_verification_does_not_leak_order_facts_or_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(1, "Order number 12345678.", expected_decision="clarify"),
                    row(
                        2,
                        "Last four digits are 1111.",
                        expected_decision="clarify",
                        expected_tool_calls="lookup_order(order_id=12345678, verification_value=1111)",
                    ),
                ],
            )
            results, _ = evaluate_dataset(path, data_dir=self.write_tool_data(directory))

        serialized = json.dumps(results)
        self.assertFalse(results[1]["privacy_violation"])
        self.assertNotIn("1111", serialized)
        self.assertNotIn("560001", serialized)
        self.assertNotIn("Shipped", serialized)
        self.assertEqual(results[1]["actual_decision"], "clarify")

    def test_failed_tool_result_with_private_fields_is_not_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(1, "Order number 12345678.", expected_decision="clarify"),
                    row(2, "Last four digits are 1111.", expected_decision="clarify"),
                ],
            )
            results, _ = evaluate_dataset(
                path,
                data_dir=self.write_tool_data(directory),
                lookup_tool=lambda _order_id, _proof: {
                    "verified": False,
                    "verification_required": True,
                    "delivery_pincode": "560001",
                    "status": "Shipped",
                    "customer_id": "C-PRIVATE",
                },
            )

        self.assertFalse(results[1]["privacy_violation"])
        serialized = json.dumps(results)
        for private in ("560001", "Shipped", "C-PRIVATE", "1111"):
            self.assertNotIn(private, serialized)

    def test_ticket_failure_duplicate_prevention_and_no_handoff_claim(self):
        ticket = Mock(return_value={"created": False, "error": "fixture failure"})
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(
                        1,
                        "Can I speak to a human?",
                        expected_decision="escalate",
                        expected_tool_calls="create_ticket()",
                    ),
                    row(2, "Can I speak to a human?", expected_decision="escalate"),
                ],
            )

            def responder(_customer, _context):
                return AgentReply(
                    "I can request a support ticket.",
                    decision="escalate",
                    required_action="human handoff requested",
                )

            results, _ = evaluate_dataset(
                path,
                responder=responder,
                data_dir=self.write_tool_data(directory),
                ticket_creator=ticket,
            )

        ticket.assert_called_once()
        first_call = json.loads(results[0]["actual_tool_calls"])
        self.assertEqual(first_call[0]["created"], False)
        self.assertFalse(results[0]["ticket_created"])
        self.assertEqual(json.loads(results[1]["actual_tool_calls"]), [])
        self.assertFalse(results[0]["handoff_performed"])
        self.assertFalse(results[1]["handoff_performed"])

    def test_success_is_counted_only_for_created_true_and_no_live_tool_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(
                directory,
                [
                    row(
                        1,
                        "Can I speak to a human?",
                        expected_decision="escalate",
                        expected_tool_calls="create_ticket()",
                    )
                ],
            )
            results, _ = evaluate_dataset(
                path,
                responder=lambda _customer, _context: AgentReply(
                    "I can request a ticket.",
                    decision="escalate",
                    required_action="human handoff requested",
                ),
                data_dir=self.write_tool_data(directory),
            )

        self.assertTrue(results[0]["ticket_created"])
        self.assertFalse(results[0]["handoff_performed"])


if __name__ == "__main__":
    unittest.main()
