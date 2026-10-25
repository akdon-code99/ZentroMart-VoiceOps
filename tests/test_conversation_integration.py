import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from conversation_agent import RAGConversationResponder
from conversation_evaluator import load_evaluation_dataset
from conversation_runner import run_conversations
from knowledge import load_knowledge_records
from rag import answer_with_evidence
from retriever import EvidenceRetriever
from vector_index import VectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROOF_VALUES = {"4533", "4265", "1581", "1266", "3357"}


class FixedQueryEncoder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            current = re.search(
                r"Current customer question:\s*([^\n]+)",
                text,
                re.IGNORECASE,
            )
            query = current.group(1) if current else text
            folded = query.casefold()
            if "coupon" in folded or "कूपन" in folded:
                category = 0
            elif "ship" in folded or "ट्रैक" in folded:
                category = 1
            elif "refund" in folded or "रिफंड" in folded or "रिफ़ंड" in folded:
                category = 2
            elif "return" in folded or "वापस" in folded or "रिटर्न" in folded:
                category = 3
            elif "voucher" in folded or "रुपये" in folded:
                category = 4
            elif "damage" in folded or "replacement" in folded or "इंसान" in folded:
                category = 5
            elif "verification proof redacted" in folded:
                category = 7
            elif "hello" in folded or "नमस्ते" in folded:
                category = 6
            else:
                category = 7
            vector = np.zeros(8, dtype=np.float32)
            vector[category] = 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def _record_vector(record):
    section = str(record.metadata.get("section_id", ""))
    category_by_section = {
        "CPN-1": 0,
        "TRK-1": 1,
        "REF-2": 2,
        "RET-7": 2,
        "PLS-2": 2,
        "RET-2": 3,
        "DEL-4": 4,
        "ESC-4": 4,
        "DWM-1": 5,
        "DWM-2": 5,
        "DWM-3": 5,
        "ESC-1": 5,
        "ESC-2": 5,
        "IDV-1": 7,
        "IDV-2": 7,
        "IDV-3": 7,
        "IDV-4": 7,
        "IDV-5": 7,
        "STY-1": 6,
        "STY-2": 6,
        "STY-3": 6,
        "STY-4": 6,
        "STY-5": 6,
    }
    row = record.metadata.get("row")
    if isinstance(row, dict) and record.metadata.get("csv_table") == "products":
        return np.asarray([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    vector = np.zeros(8, dtype=np.float32)
    category = category_by_section.get(section, 7)
    vector[category] = 1.0
    return vector


def _production_retriever():
    records = load_knowledge_records(
        PROJECT_ROOT,
        token_counter=lambda text: len(text.split()),
    )
    vectors = np.asarray([_record_vector(record) for record in records])
    index = VectorIndex(
        records=records,
        vectors=vectors,
        manifest={"record_count": len(records), "vector_dimensions": 8},
    )
    return EvidenceRetriever(index, FixedQueryEncoder())


def _chat_content(messages):
    prompt = messages[1]["content"]
    match = re.search(
        r"Current customer question:\s*(.*?)"
        r"\n\n(?:Verified order facts|Retrieved evidence)",
        prompt,
        re.DOTALL,
    )
    question = (
        match.group(1).strip().splitlines()[-1]
        if match
        else prompt.split("\n\nRetrieved evidence", maxsplit=1)[0]
    )
    lowered = question.casefold()

    if "verification proof redacted" in lowered:
        answer = "The caller is verified [IDV-2]."
    elif "hello" in lowered or "नमस्ते" in lowered:
        answer = "Hello! I am ZentroMart's AI customer-support assistant [STY-1]."
    elif "coupon" in lowered or "कूपन" in lowered:
        answer = "You can use one coupon per order [CPN-1]."
    elif "refund" in lowered or "रिफ़ंड" in lowered or "रिफंड" in lowered:
        answer = (
            "Wallet refunds are instant or within 2 hours; other applicable "
            "refunds take 5 to 7 days [REF-2] [RET-7] [PLS-2]."
        )
    elif "return" in lowered or "वापस" in lowered or "रिटर्न" in lowered:
        answer = "The return deadline is 2026-10-27 [RET-2]."
    elif "ship" in lowered:
        answer = "Your order shipped on 2026-10-18 [TRK-1]."
    elif "voucher" in lowered:
        answer = "I cannot approve a ₹300 voucher under the policy [DEL-4] [ESC-4]."
    elif "person" in lowered or "इंसान" in lowered:
        answer = "I will create a ticket for colleague review [ESC-1]."
    else:
        answer = "I can help with that [STY-1]."
    return json.dumps(
        {"kind": "factual_answer", "answer": answer},
        ensure_ascii=False,
    )


def _rag_chat_mock(messages):
    return SimpleNamespace(message=SimpleNamespace(content=_chat_content(messages)))


class ConversationProductionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_evaluation_dataset()
        cls.retriever = _production_retriever()

    def _messages(self, conversation_id):
        return [
            turn.message
            for turn in self.dataset[conversation_id]
        ]

    def _production_responder(self):
        return RAGConversationResponder(
            self.retriever,
            rag_answerer=answer_with_evidence,
        )

    def _run(self, conversation_ids, lookup, ticket):
        messages = [
            message
            for conversation_id in conversation_ids
            for message in self._messages(conversation_id)
        ]
        responder = self._production_responder()
        with patch("rag.chat", side_effect=_rag_chat_mock) as chat:
            run = run_conversations(
                messages,
                responder,
                lookup_order_fn=lookup,
                create_ticket_fn=ticket,
            )
        return run, chat

    def test_successful_verification_policy_only_topic_switch_and_mixed_followup(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {
                    "found": True,
                    "verified": True,
                    "status": "shipped",
                    "shipped_on": "2026-10-18",
                    "payment_method": "wallet",
                    "delivered_on": "2026-10-12",
                    "return_status": "eligible",
                    "is_plus": True,
                    "items": [
                        {
                            "sku": "FAS-1100",
                            "title": "Kaafi Running Sneakers",
                            "return_category": "fashion",
                            "qty": 1,
                        }
                    ],
                    "phone": "+91 5123454533",
                    "delivery_pincode": "560001",
                    "customer_id": "PRIVATE-CUSTOMER",
                    "email": "private@example.com",
                },
                {"verification_required": True},
                {
                    "found": True,
                    "verified": True,
                    "status": "shipped",
                    "shipped_on": "2026-10-18",
                    "phone": "+91 5123459260",
                    "delivery_pincode": "560001",
                },
            ]
        )
        ticket = Mock(return_value={"created": False, "error": "unused"})

        run, chat = self._run(["PC003", "PC013"], lookup, ticket)
        results = {
            (result.conversation_id, result.turn): result
            for result in run.results
        }

        self.assertEqual(run.states["PC003"].verification_status, "verified")
        self.assertEqual(results[("PC003", 4)].decision, "answer")
        self.assertIn("2026-10-27", results[("PC003", 4)].response)
        self.assertIn("RET-2", results[("PC003", 4)].evidence_ids)
        self.assertIn("2 hours", results[("PC003", 5)].response)
        self.assertIn("5 to 7 days", results[("PC003", 5)].response)
        self.assertEqual(results[("PC013", 4)].decision, "answer")
        self.assertIn("CPN-1", results[("PC013", 4)].evidence_ids)
        self.assertIn("TRK-1", results[("PC013", 5)].evidence_ids)
        self.assertIn(
            "shipped_on: 2026-10-18",
            "\n".join(call.args[0][1]["content"] for call in chat.call_args_list),
        )
        safe_transcript = repr(run) + repr(chat.call_args_list)
        for private in ("4533", "560001", "PRIVATE-CUSTOMER", "private@example.com"):
            self.assertNotIn(private, safe_transcript)
        ticket.assert_not_called()

    def test_failed_verification_never_calls_rag_with_proof_or_private_facts(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {"verified": False, "verification_required": True},
                {"verified": False, "verification_required": True},
            ]
        )
        ticket = Mock(return_value={"created": True, "ticket_id": "TKT-TEST"})

        run, chat = self._run(["PC021"], lookup, ticket)

        self.assertEqual(lookup.call_count, 3)
        self.assertEqual(run.states["PC021"].verification_status, "exhausted")
        self.assertEqual(run.results[-1].decision, "refuse")
        self.assertNotIn("1266", repr(run))
        self.assertNotIn("1266", repr(chat.call_args_list))
        self.assertNotIn("private", repr(run).casefold())
        ticket.assert_not_called()

    def test_refusal_and_human_ticket_action_are_real_rag_runner_behavior(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {"found": True, "verified": True, "status": "delivered"},
                {"verification_required": True},
                {"found": True, "verified": True, "status": "delivered"},
            ]
        )
        ticket = Mock(return_value={"created": True, "ticket_id": "TKT-MOCK-1"})

        run, _chat = self._run(["PC015", "PC005"], lookup, ticket)
        results = {
            (result.conversation_id, result.turn): result
            for result in run.results
        }

        refused = results[("PC015", 4)]
        self.assertEqual(refused.decision, "refuse")
        self.assertIn("cannot approve", refused.response.casefold())
        ticket_response = results[("PC005", 4)]
        self.assertEqual(ticket_response.decision, "escalate")
        self.assertTrue(ticket_response.ticket_created)
        self.assertFalse(ticket_response.handoff_performed)
        self.assertIn("TKT-MOCK-1", ticket_response.response)
        self.assertIn("have not transferred you", ticket_response.response)
        self.assertEqual(ticket.call_count, 1)
        self.assertNotIn("1581", repr(run))

    def test_high_value_damage_uses_production_deterministic_policy_path(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {
                    "found": True,
                    "verified": True,
                    "status": "delivered",
                    "damage_sku": "ELE-0020",
                    "damage_status": "reported",
                    "items": [
                        {
                            "sku": "ELE-0020",
                            "title": "Zentro Vista 43-inch 4K Smart TV, Beige",
                            "qty": 1,
                            "unit_price_inr": 31809,
                        }
                    ],
                    "phone": "+91 5123454533",
                    "delivery_pincode": "560001",
                },
            ]
        )
        ticket = Mock(return_value={"created": True, "ticket_id": "TKT-DAMAGE-1"})

        run, chat = self._run(["PC002"], lookup, ticket)
        damage_result = run.results[-1]

        self.assertEqual(damage_result.decision, "escalate")
        self.assertTrue(damage_result.ticket_created)
        self.assertFalse(damage_result.handoff_performed)
        for evidence_id in ("DWM-3", "ESC-1", "ESC-2", "products.csv:ELE-0020"):
            self.assertIn(evidence_id, damage_result.evidence_ids)
        self.assertIn("Priority Resolution", damage_result.response)
        self.assertIn("TKT-DAMAGE-1", damage_result.response)
        self.assertIn("have not transferred you", damage_result.response)
        self.assertEqual(chat.call_count, 2)
        ticket.assert_called_once()
        self.assertNotIn("4533", repr(run))
        self.assertNotIn("560001", repr(run))

    def test_llm_false_ticket_claim_only_becomes_ticket_after_mock_tool_success(self):
        lookup = Mock(
            side_effect=[
                {"verification_required": True},
                {"found": True, "verified": True, "status": "delivered"},
            ]
        )
        ticket = Mock(return_value={"created": False, "error": "mock failure"})
        run, _chat = self._run(["PC005"], lookup, ticket)

        result = run.results[-1]
        self.assertEqual(result.decision, "escalate")
        self.assertFalse(result.ticket_created)
        self.assertFalse(result.handoff_performed)
        self.assertIn("could not create", result.response.casefold())
        self.assertIn("no human handoff", result.response.casefold())
        ticket.assert_called_once()
        self.assertEqual(lookup.call_count, 2)


if __name__ == "__main__":
    unittest.main()
