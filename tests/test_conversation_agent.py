import unittest
from unittest.mock import Mock

from conversation_agent import RAGConversationResponder
from conversation_runner import ConversationMessage, run_conversations
from rag import RAGAnswer
from retriever import RetrievalHit


def message(turn, text):
    return ConversationMessage(
        conversation_id="PC001",
        turn=turn,
        language="en",
        customer_message=text,
    )


def hit(section_id):
    return RetrievalHit(
        evidence_id=section_id,
        text=f"Policy evidence for {section_id}.",
        source_text=f"## [{section_id}]",
        metadata={"section_id": section_id, "status": "current"},
        score=0.9,
    )


def answer(
    text="Answer [RET-2].",
    *,
    decision="answer",
    required_action="",
    evidence=(),
    grounding_status="model_generated",
):
    return RAGAnswer(
        answer=text,
        evidence=list(evidence),
        required_action=required_action,
        grounding_status=grounding_status,
        decision=decision,
    )


class ConversationAgentTests(unittest.TestCase):
    def test_policy_only_question_uses_rag_without_order_lookup(self):
        rag = Mock(return_value=answer(evidence=[hit("RET-2")]))
        responder = RAGConversationResponder(object(), rag_answerer=rag)
        lookup = Mock()

        run = run_conversations(
            [message(1, "What is the return window?")],
            responder,
            lookup_order_fn=lookup,
        )

        lookup.assert_not_called()
        rag.assert_called_once()
        self.assertIn("return window", rag.call_args.args[0])
        self.assertEqual(run.results[0].decision, "answer")
        self.assertEqual(run.results[0].evidence_ids, ("RET-2",))

    def test_general_policy_question_does_not_lookup_order_id(self):
        rag = Mock(return_value=answer(evidence=[hit("CPN-1")]))
        responder = RAGConversationResponder(object(), rag_answerer=rag)
        lookup = Mock()

        run = run_conversations(
            [message(1, "Order number 12345678; can I use two coupons next time?")],
            responder,
            lookup_order_fn=lookup,
        )

        lookup.assert_not_called()
        rag.assert_called_once()
        self.assertEqual(run.results[0].decision, "answer")

    def test_verified_order_question_receives_only_relevant_sanitized_facts(self):
        private_tool_result = {
            "found": True,
            "verified": True,
            "status": "shipped",
            "shipped_on": "2026-10-20",
            "delivery_pincode": "560001",
            "phone": "+91 5123456789",
            "email": "private@example.com",
            "customer_id": "C123",
            "address": "private address",
            "items": [
                {
                    "sku": "ELE-0012",
                    "title": "Zentro Laptop",
                    "qty": 1,
                    "unit_price_inr": 25000,
                    "customer_note": "private item detail",
                }
            ],
        }
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                private_tool_result,
            ]
        )
        rag = Mock(return_value=answer(evidence=[hit("TRK-1")]))
        responder = RAGConversationResponder(object(), rag_answerer=rag)

        run = run_conversations(
            [
                message(1, "Order number 12345678."),
                message(2, "Last four digits 4533."),
                message(3, "Where is my order?"),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        facts = rag.call_args.kwargs["verified_order_facts"]
        self.assertEqual(facts, {"status": "shipped", "shipped_on": "2026-10-20"})
        prompt_text = repr(rag.call_args)
        for private_value in (
            "560001",
            "+91 5123456789",
            "private@example.com",
            "C123",
            "private address",
            "customer_note",
            "4533",
        ):
            self.assertNotIn(private_value, prompt_text)
        self.assertEqual(run.states["PC001"].verification_status, "verified")

    def test_mixed_question_passes_verified_facts_and_policy_to_rag(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {
                    "found": True,
                    "verified": True,
                    "delivered_on": "2026-10-20",
                    "payment_method": "upi",
                    "return_status": "not_started",
                    "items": [{"title": "Kaafi Running Sneakers", "sku": "FAS-1100"}],
                },
            ]
        )
        rag = Mock(return_value=answer("Your item is within the return window [RET-2].",
                                       evidence=[hit("RET-2")]))
        responder = RAGConversationResponder(object(), rag_answerer=rag)

        run_conversations(
            [
                message(1, "Order number 12345678."),
                message(2, "Last four digits 4533."),
                message(3, "Can I return this item?"),
            ],
            responder,
            lookup_order_fn=lookup,
        )

        self.assertEqual(rag.call_count, 2)
        self.assertIn("Can I return this item?", rag.call_args_list[-1].args[0])
        self.assertEqual(
            rag.call_args_list[-1].kwargs["verified_order_facts"],
            {
                "delivered_on": "2026-10-20",
                "return_status": "not_started",
                "items": [{"title": "Kaafi Running Sneakers", "sku": "FAS-1100"}],
            },
        )

    def test_refusal_decision_is_preserved(self):
        rag = Mock(
            return_value=answer(
                "This order is not eligible for cancellation [CAN-2].",
                decision="refuse",
                evidence=[hit("CAN-2")],
            )
        )
        responder = RAGConversationResponder(object(), rag_answerer=rag)

        run = run_conversations([message(1, "Cancel my order.")], responder)

        self.assertEqual(run.results[0].decision, "refuse")
        self.assertIn("[CAN-2]", run.results[0].response)

    def test_clarification_decision_is_preserved(self):
        rag = Mock(
            return_value=answer(
                "Which payment method did you use?",
                decision="clarify",
                grounding_status="insufficient_required_evidence",
            )
        )
        responder = RAGConversationResponder(object(), rag_answerer=rag)

        run = run_conversations([message(1, "Where is my refund?")], responder)

        self.assertEqual(run.results[0].decision, "clarify")
        self.assertEqual(run.results[0].response, "Which payment method did you use?")

    def test_escalation_ticket_failure_is_honest(self):
        rag = Mock(
            return_value=answer(
                "This needs human review [ESC-1].",
                required_action="create_ticket_then_human_handoff",
                evidence=[hit("ESC-1"), hit("ESC-2")],
            )
        )
        responder = RAGConversationResponder(object(), rag_answerer=rag)
        create_ticket = Mock(return_value={"created": False, "error": "failed"})

        run = run_conversations(
            [message(1, "A damaged item needs human review.")],
            responder,
            create_ticket_fn=create_ticket,
        )

        create_ticket.assert_called_once()
        self.assertEqual(run.results[0].decision, "escalate")
        self.assertFalse(run.results[0].ticket_created)
        self.assertFalse(run.results[0].handoff_performed)
        self.assertIn("could not create", run.results[0].response)
        self.assertIn("no human handoff", run.results[0].response)

    def test_successful_ticket_creation_does_not_claim_handoff(self):
        rag = Mock(
            return_value=answer(
                "This issue requires escalation [ESC-2].",
                required_action="create_ticket_then_human_handoff",
                evidence=[hit("ESC-2")],
            )
        )
        responder = RAGConversationResponder(object(), rag_answerer=rag)
        create_ticket = Mock(return_value={"created": True, "ticket_id": "TKT001"})

        run = run_conversations(
            [message(1, "Please escalate this damaged item.")],
            responder,
            create_ticket_fn=create_ticket,
        )

        self.assertTrue(run.results[0].ticket_created)
        self.assertFalse(run.results[0].handoff_performed)
        self.assertIn("TKT001", run.results[0].response)
        self.assertIn("have not transferred you", run.results[0].response)

    def test_duplicate_ticket_action_is_not_executed_twice(self):
        rag = Mock(
            return_value=answer(
                "This issue needs a ticket [ESC-2].",
                required_action="create_ticket_then_human_handoff",
                evidence=[hit("ESC-2")],
            )
        )
        responder = RAGConversationResponder(object(), rag_answerer=rag)
        create_ticket = Mock(return_value={"created": True, "ticket_id": "TKT001"})

        run = run_conversations(
            [
                message(1, "Please escalate this damaged laptop."),
                message(2, "Can you escalate the damaged laptop again?"),
            ],
            responder,
            create_ticket_fn=create_ticket,
        )

        create_ticket.assert_called_once()
        self.assertIn("already created", run.results[1].response)
        self.assertTrue(run.results[1].ticket_created)

    def test_distinct_required_issue_can_create_a_new_ticket(self):
        rag = Mock(
            return_value=answer(
                "This issue needs a ticket [ESC-2].",
                required_action="create_ticket_then_human_handoff",
                evidence=[hit("ESC-2")],
            )
        )
        responder = RAGConversationResponder(object(), rag_answerer=rag)
        create_ticket = Mock(
            side_effect=[
                {"created": True, "ticket_id": "TKT001"},
                {"created": True, "ticket_id": "TKT002"},
            ]
        )

        run = run_conversations(
            [
                message(1, "A damaged laptop needs escalation."),
                message(2, "I also have an overdue refund issue."),
            ],
            responder,
            create_ticket_fn=create_ticket,
        )

        self.assertEqual(create_ticket.call_count, 2)
        self.assertIn("TKT001", run.results[0].response)
        self.assertIn("TKT002", run.results[1].response)

    def test_ticket_summary_and_rag_context_never_contain_verification_proof(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {
                    "found": True,
                    "verified": True,
                    "status": "delivered",
                },
            ]
        )
        rag = Mock(
            return_value=answer(
                "Escalation is required [ESC-2].",
                required_action="create_ticket_then_human_handoff",
                evidence=[hit("ESC-2")],
            )
        )
        responder = RAGConversationResponder(object(), rag_answerer=rag)
        create_ticket = Mock(return_value={"created": True, "ticket_id": "TKT002"})

        run_conversations(
            [
                message(1, "Order number 12345678."),
                message(2, "Last four digits 4533."),
                message(3, "Please escalate this order issue."),
            ],
            responder,
            lookup_order_fn=lookup,
            create_ticket_fn=create_ticket,
        )

        summary = create_ticket.call_args.kwargs["summary"]
        self.assertNotIn("4533", summary)
        self.assertNotIn("4533", repr(rag.call_args))


if __name__ == "__main__":
    unittest.main()
