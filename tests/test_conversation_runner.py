import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from conversation_runner import (
    ConversationContext,
    ConversationMessage,
    CustomerInput,
    load_conversations,
    run_conversations,
)


def message(conversation_id, turn, text, language="en"):
    return ConversationMessage(
        conversation_id=conversation_id,
        turn=turn,
        language=language,
        customer_message=text,
    )


class ConversationRunnerTests(unittest.TestCase):
    def test_csv_rows_are_grouped_and_sorted_by_numeric_turn(self):
        messages = [
            message("PC010", 2, "second"),
            message("PC011", 1, "other conversation"),
            message("PC010", 1, "first"),
        ]
        run = run_conversations(messages, lambda customer, context: customer.customer_message)

        self.assertEqual(
            [(row.conversation_id, row.turn) for row in run.results],
            [("PC010", 1), ("PC010", 2), ("PC011", 1)],
        )
        self.assertEqual(
            [row.response for row in run.results],
            ["first", "second", "other conversation"],
        )

    def test_loader_does_not_include_evaluation_only_fields(self):
        fields = [
            "conversation_id",
            "turn",
            "language",
            "customer_message",
            "expected_decision",
            "must_include",
            "must_not_include",
            "expected_tool_calls",
            "gold_evidence",
            "notes",
        ]
        row = {
            "conversation_id": "PC001",
            "turn": "1",
            "language": "en",
            "customer_message": "Hello",
            "expected_decision": "answer",
            "must_include": "secret evaluation value",
            "must_not_include": "another evaluation value",
            "expected_tool_calls": "create_ticket()",
            "gold_evidence": "ESC-1",
            "notes": "evaluation-only note",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversations.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)

            loaded = load_conversations(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].customer_message, "Hello")
        self.assertEqual(
            set(loaded[0].__dataclass_fields__),
            {"conversation_id", "turn", "language", "customer_message"},
        )

    def test_state_is_isolated_between_conversations(self):
        seen = {}
        lookup = Mock(return_value={"found": True, "verified": True})

        def responder(customer, context):
            seen.setdefault(customer.customer_message, context)
            return f"reply to {customer.customer_message}"

        run = run_conversations(
            [
                message("PC001", 1, "Order number 12345678, last four digits 1234."),
                message("PC002", 1, "I have a different issue."),
                message("PC001", 2, "Where is it?"),
                message("PC002", 2, "Can you help?"),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        first_context = seen["Where is it?"]
        second_context = seen["Can you help?"]
        self.assertEqual(first_context.order_id, "12345678")
        self.assertEqual(first_context.verification_status, "verified")
        self.assertEqual(first_context.verification_attempts, 1)
        self.assertFalse(first_context.ticket_created)
        self.assertEqual(
            first_context.history[0].content,
            "[verification proof redacted]",
        )
        self.assertEqual(second_context.order_id, None)
        self.assertEqual(second_context.history[0].content, "I have a different issue.")
        self.assertNotIn(
            "[verification proof redacted]",
            second_context.issue_context,
        )
        self.assertIsNot(
            run.states["PC001"].history,
            run.states["PC002"].history,
        )

    def test_follow_up_receives_prior_exchange_and_order_context(self):
        contexts = []
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {"found": True, "verified": True},
            ]
        )

        def responder(customer, context):
            contexts.append((customer, context))
            return "I can help with that."

        run_conversations(
            [
                message("PC001", 1, "My order number is 12345678."),
                message("PC001", 2, "Last four digits 1234."),
                message("PC001", 3, "What about the refund?"),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        follow_up_input, follow_up_context = contexts[1]
        self.assertEqual(follow_up_input.customer_message, "What about the refund?")
        self.assertEqual(follow_up_context.order_id, "12345678")
        self.assertEqual(follow_up_context.current_topic, "What about the refund?")
        self.assertEqual(
            [(entry.role, entry.content) for entry in follow_up_context.history],
            [
                ("user", "My order number is 12345678."),
                (
                    "assistant",
                    "Please share either the registered phone’s last four digits "
                    "or the delivery pincode for this order. I can’t share order "
                    "details until verification is complete.",
                ),
                ("user", "[verification proof redacted]"),
                ("assistant", "I can help with that."),
            ],
        )
        self.assertIn("My order number is 12345678.", follow_up_context.issue_context)
        self.assertIsInstance(follow_up_context, ConversationContext)
        self.assertIsInstance(follow_up_input, CustomerInput)

    def test_successful_proof_uses_prior_order_question_for_minimal_facts(self):
        contexts = []
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {
                    "found": True,
                    "verified": True,
                    "status": "Shipped",
                    "promised_date": "2026-04-12",
                    "delivery_pincode": "560001",
                    "customer_phone": "9999994533",
                },
            ]
        )

        def responder(_customer, context):
            contexts.append((_customer, context))
            return "Your order is shipped."

        run = run_conversations(
            [
                message("PC001", 1, "Where is my order 12345678?"),
                message("PC001", 2, "Last four digits 4533."),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        proof_input, proof_context = contexts[0]
        self.assertEqual(
            proof_input.customer_message,
            "[verification proof redacted]",
        )
        self.assertEqual(proof_context.verified_order_facts["status"], "Shipped")
        self.assertEqual(
            proof_context.verified_order_facts["promised_date"],
            "2026-04-12",
        )
        self.assertNotIn("delivery_pincode", proof_context.verified_order_facts)
        self.assertNotIn("customer_phone", proof_context.verified_order_facts)
        self.assertNotIn("4533", repr(run))

    def test_topic_switch_updates_topic_without_discarding_prior_issue(self):
        contexts = []
        lookup = Mock()

        def responder(customer, context):
            contexts.append(context)
            return "Noted."

        run_conversations(
            [
                message("PC001", 1, "When will my order ship?"),
                message("PC001", 2, "Can I use two coupons next time?"),
                message("PC001", 3, "And when was my order shipped?"),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        resumed_topic = contexts[2]
        self.assertEqual(
            resumed_topic.current_topic,
            "And when was my order shipped?",
        )
        self.assertEqual(
            resumed_topic.issue_context,
            (
                "When will my order ship?",
                "Can I use two coupons next time?",
                "And when was my order shipped?",
            ),
        )
        self.assertEqual(resumed_topic.history[-1].content, "Noted.")
        lookup.assert_not_called()

    def test_missing_proof_requests_one_verification_factor(self):
        lookup = Mock(return_value={"verification_required": True})
        responder = Mock(return_value="should not answer before verification")

        run = run_conversations(
            [message("PC001", 1, "My order number is 12345678.")],
            responder,
            lookup_order_fn=lookup,
        )

        lookup.assert_called_once_with("12345678")
        responder.assert_not_called()
        self.assertIn("last four digits", run.results[0].response)
        self.assertIn("delivery pincode", run.results[0].response)
        self.assertEqual(run.states["PC001"].verification_status, "pending")
        self.assertEqual(run.states["PC001"].verification_attempts, 0)

    def test_success_requires_tool_confirmation_and_redacts_proof(self):
        private_result = {
            "found": True,
            "verified": True,
            "delivery_pincode": "560001",
            "total_inr": 12345,
            "customer_first_name": "PrivateName",
        }
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                private_result,
            ]
        )
        received = []

        def responder(customer, context):
            received.append((customer, context))
            return "Verification complete."

        run = run_conversations(
            [
                message("PC001", 1, "My order number is 12345678."),
                message("PC001", 2, "Last four digits 4533 hain."),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        self.assertEqual(
            lookup.call_args_list[1].args,
            ("12345678",),
        )
        self.assertEqual(
            lookup.call_args_list[1].kwargs,
            {"verification_value": "4533"},
        )
        self.assertEqual(run.states["PC001"].verification_status, "verified")
        self.assertEqual(run.states["PC001"].verification_attempts, 1)
        self.assertEqual(received[0][1].verification_status, "verified")
        self.assertEqual(received[0][0].customer_message, "[verification proof redacted]")
        self.assertNotIn("4533", repr(received))
        self.assertNotIn("4533", repr(run.states["PC001"].history))
        self.assertNotIn("4533", repr(run.states["PC001"].issue_context))
        self.assertNotIn("4533", repr(run.results))
        self.assertNotIn("560001", repr(received))
        self.assertNotIn("PrivateName", repr(run.results))

    def test_invalid_proof_does_not_reveal_tool_result_details(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {
                    "verified": False,
                    "verification_required": True,
                    "delivery_pincode": "560001",
                    "customer_first_name": "PrivateName",
                },
            ]
        )
        responder = Mock(return_value="should not answer")

        run = run_conversations(
            [
                message("PC001", 1, "Order number 12345678."),
                message("PC001", 2, "My pincode is 987654."),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        self.assertEqual(run.states["PC001"].verification_attempts, 1)
        self.assertEqual(run.states["PC001"].verification_status, "pending")
        self.assertIn("try once more", run.results[1].response)
        self.assertNotIn("560001", repr(run))
        self.assertNotIn("PrivateName", repr(run))
        self.assertNotIn("987654", repr(run))
        responder.assert_not_called()

    def test_bare_proof_reply_is_used_only_while_verification_is_pending(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {"found": True, "verified": True},
            ]
        )
        received = []

        def responder(customer, context):
            received.append(customer.customer_message)
            return "Verified."

        run = run_conversations(
            [
                message("PC001", 1, "Order number 12345678."),
                message("PC001", 2, "4533"),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        lookup.assert_called_with(
            "12345678",
            verification_value="4533",
        )
        self.assertEqual(received, ["[verification proof redacted]"])
        self.assertNotIn("4533", repr(run))

    def test_failed_attempt_limit_cannot_be_reset_by_topic_or_rephrasing(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {"verified": False, "verification_required": True},
                {"verified": False, "verification_required": True},
            ]
        )
        responder = Mock(return_value="should not answer")
        run = run_conversations(
            [
                message("PC001", 1, "Order number 12345678."),
                message("PC001", 2, "Last four digits 1111."),
                message("PC001", 3, "Different question: can I use two coupons next time?"),
                message("PC001", 4, "Actually, my last four digits are 2222."),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        self.assertEqual(lookup.call_count, 3)
        self.assertEqual(run.states["PC001"].verification_attempts, 2)
        self.assertEqual(run.states["PC001"].verification_status, "exhausted")
        self.assertEqual(run.results[3].decision, "refuse")
        self.assertIn("couldn’t verify", run.results[3].response)
        self.assertNotIn("12345678", run.results[3].response)
        self.assertNotIn("1111", repr(run))
        self.assertNotIn("2222", repr(run))
        responder.assert_called_once()

    def test_ticket_state_cannot_verify_caller(self):
        state = run_conversations(
            [message("PC001", 1, "Order number 12345678.")],
            Mock(),
            lookup_order_fn=Mock(return_value={"verification_required": True}),
        ).states["PC001"]

        self.assertFalse(state.ticket_created)
        self.assertEqual(state.verification_status, "pending")


if __name__ == "__main__":
    unittest.main()
