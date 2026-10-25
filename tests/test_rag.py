import json
import unittest
from unittest.mock import patch

from knowledge import EvidenceRecord
from rag import _required_citations, answer_with_evidence
from retriever import EvidenceRetriever, RetrievalHit


class FakeRetriever:
    def __init__(self):
        self.query = None
        self.search_options = None

    def search(self, query, top_k, **options):
        self.query = query
        self.search_options = {"top_k": top_k, **options}
        return [
            RetrievalHit(
                evidence_id="REF-2",
                text="Wallet refunds are instant, within 2 hours.",
                source_text="## [REF-2] Refund timelines",
                metadata={
                    "source_path": "kb/pol_refunds.md",
                    "doc_id": "POL-REF",
                    "doc_type": "policy",
                    "status": "current",
                    "version": "2.1",
                    "section_id": "REF-2",
                },
                score=0.9,
            )
        ]

    def authority_guidance(self):
        return "Current policy outranks FAQs and marketing."


class EvidenceRetriever(FakeRetriever):
    def __init__(self, evidence):
        super().__init__()
        self.evidence = evidence

    def search(self, query, top_k, **options):
        self.query = query
        self.search_options = {"top_k": top_k, **options}
        return self.evidence


def hit(evidence_id, text, metadata):
    return RetrievalHit(evidence_id, text, text, metadata, 0.9)


def model_response(answer=None, *, kind="factual_answer"):
    payload = {"kind": kind}
    if answer is not None:
        payload["answer"] = answer
    return json.dumps(payload, ensure_ascii=False)


class RAGAnswerTests(unittest.TestCase):
    @patch("rag.chat")
    def test_passes_retrieved_source_and_authority_but_not_gold_answers(self, mock_chat):
        mock_chat.return_value.message.content = model_response(
            "Wallet refunds are instant [REF-2]."
        )
        retriever = FakeRetriever()

        result = answer_with_evidence(
            "How fast are wallet refunds?",
            "en",
            retriever,
            top_k=5,
        )

        self.assertEqual(result.answer, "Wallet refunds are instant [REF-2].")
        self.assertEqual(retriever.query, "How fast are wallet refunds?")
        self.assertEqual(
            retriever.search_options,
            {
                "top_k": 5,
                "candidate_pool": 16,
                "authority_adjustment": 0.03,
                "relevance_window": 0.002,
                "max_product_rows": 2,
            },
        )
        messages = mock_chat.call_args.args[0]
        self.assertIn("Current policy outranks FAQs", messages[0]["content"])
        self.assertIn("REF-2", messages[1]["content"])
        self.assertIn("Evidence role: retrieved evidence", messages[1]["content"])
        self.assertIn("Wallet refunds are instant", messages[1]["content"])

    @patch("rag.chat")
    def test_verified_order_context_is_separate_from_policy_evidence(self, mock_chat):
        mock_chat.return_value.message.content = model_response(
            "Your item is within the return window [RET-2]."
        )
        retriever = EvidenceRetriever(
            [
                hit(
                    "RET-2",
                    "Eligible items can be returned within the listed window.",
                    {"section_id": "RET-2", "status": "current"},
                )
            ]
        )

        result = answer_with_evidence(
            "Can I return this item?",
            "en",
            retriever,
            verified_order_facts={
                "delivered_on": "2026-10-20",
                "items": [{"sku": "FAS-1100", "title": "Kaafi Running Sneakers"}],
            },
        )

        self.assertEqual(result.answer, "Your item is within the return window [RET-2].")
        self.assertIn("FAS-1100", retriever.query)
        user_prompt = mock_chat.call_args.args[0][1]["content"]
        self.assertIn("Verified order facts", user_prompt)
        self.assertIn("2026-10-20", user_prompt)
        self.assertIn("RET-2", user_prompt)

    @patch("rag.chat")
    def test_policy_refusal_is_structured_without_changing_cited_answer(self, mock_chat):
        mock_chat.return_value.message.content = model_response(
            "This order is not eligible for cancellation [CAN-2]."
        )
        evidence = [
            hit(
                "CAN-2",
                "Shipped orders cannot be cancelled.",
                {"section_id": "CAN-2", "status": "current"},
            )
        ]

        result = answer_with_evidence(
            "Can I cancel a shipped order?",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(result.decision, "refuse")
        self.assertIn("[CAN-2]", result.answer)

    @patch("rag.chat")
    def test_single_clarifying_question_is_structured_as_clarify(self, mock_chat):
        mock_chat.return_value.message.content = model_response(
            "Which payment method did you use? [REF-2]"
        )
        evidence = [
            hit(
                "REF-2",
                "Refund times depend on the payment method.",
                {"section_id": "REF-2", "status": "current"},
            )
        ]

        result = answer_with_evidence(
            "Where is my refund?",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(result.decision, "clarify")
        self.assertIn("Which payment method", result.answer)

    @patch("rag.chat")
    def test_structured_abstention_for_unsupported_practice_questions(
        self,
        mock_chat,
    ):
        mock_chat.return_value.message.content = model_response(kind="abstention")
        questions = (
            ("PQ0004", "Can I schedule a delivery at midnight?", "en"),
            ("PQ0009", "Is there a student discount?", "en"),
            ("PQ0015", "Bulk order pe GST invoice milega kya?", "hi-en"),
            ("PQ0037", "क्या आपकी कोई दुकान कोच्चि में है?", "hi"),
            ("PQ0053", "Can I pick up my order from a store near me?", "en"),
        )

        for question_id, question, language in questions:
            with self.subTest(question_id=question_id):
                result = answer_with_evidence(
                    question,
                    language,
                    FakeRetriever(),
                )

                self.assertNotEqual(result.answer, mock_chat.return_value.message.content)
                self.assertEqual(result.decision, "clarify")
                self.assertEqual(result.grounding_status, "model_abstention")
                if language == "hi":
                    self.assertIn("पुष्टि नहीं कर सकता", result.answer)
                elif language == "hi-en":
                    self.assertIn("pushti nahi kar sakta", result.answer)
                else:
                    self.assertIn("cannot confirm", result.answer)

    @patch("rag.chat")
    def test_invalid_structured_responses_fail_closed(self, mock_chat):
        invalid_responses = (
            "not JSON",
            "[]",
            "NaN",
            '{"kind":"abstention","kind":"factual_answer","answer":"claim [REF-2]"}',
            '{"kind":"abstention"} trailing text',
            '{"kind":"abstention","answer":"Wallet refunds are instant."}',
            '{"kind":"factual_answer"}',
            '{"answer":"Wallet refunds are instant [REF-2]."}',
            '{"kind":"factual_answer","answer":42}',
            '{"kind":"factual_answer","answer":"  "}',
            '{"kind":"unsupported"}',
        )
        for content in invalid_responses:
            with self.subTest(content=content):
                mock_chat.return_value.message.content = content
                result = answer_with_evidence(
                    "How fast are wallet refunds?",
                    "en",
                    FakeRetriever(),
                )
    
                self.assertIn("cannot confirm", result.answer)
                self.assertEqual(
                    result.grounding_status,
                    "invalid_structured_response",
                )
                self.assertEqual(result.decision, "clarify")


    @patch("rag.chat")
    def test_factual_answer_without_citations_fails_validation(self, mock_chat):
        mock_chat.return_value.message.content = model_response(
            "Wallet refunds are instant."
        )

        result = answer_with_evidence(
            "How fast are wallet refunds?",
            "en",
            FakeRetriever(),
        )

        self.assertEqual(result.grounding_status, "citation_validation_failed")
        self.assertIn("cannot confirm", result.answer)

    @patch("rag.chat")
    def test_mixed_abstention_and_factual_text_is_validated_as_factual(
        self,
        mock_chat,
    ):
        mock_chat.return_value.message.content = model_response(
            "I cannot confirm this, but wallet refunds are instant."
        )

        result = answer_with_evidence(
            "How fast are wallet refunds?",
            "en",
            FakeRetriever(),
        )

        self.assertEqual(result.grounding_status, "citation_validation_failed")
        self.assertIn("cannot confirm", result.answer)

    @patch("rag.chat")
    def test_labels_historical_evidence_and_keeps_successor_governing(self, mock_chat):
        mock_chat.return_value.message.content = model_response(
            "The applicable historical window was 10 days [RETv3-2]."
        )

        class HistoricalRetriever(FakeRetriever):
            def search(self, query, top_k, **options):
                return [
                    RetrievalHit(
                        evidence_id="RET-3",
                        text="Electronics delivered before 1 August keep the older window.",
                        source_text="RET-3",
                        metadata={
                            "doc_type": "policy",
                            "status": "current",
                            "retrieval_role": "governing_current_policy",
                        },
                        score=0.9,
                    ),
                    RetrievalHit(
                        evidence_id="RETv3-2",
                        text="[HISTORICAL POLICY REFERENCE — NOT GOVERNING CURRENT POLICY]\n"
                        "Electronics return window: 10 days.",
                        source_text="RETv3-2",
                        metadata={
                            "doc_type": "policy",
                            "status": "superseded",
                            "retrieval_role": "historical_reference",
                        },
                        score=0.8,
                    ),
                ]

        answer_with_evidence(
            "My smartwatch was delivered on 20 July. What was its return window?",
            "en",
            HistoricalRetriever(),
        )

        messages = mock_chat.call_args.args[0]
        self.assertIn("current successor remains governing", messages[0]["content"])
        self.assertIn("Evidence role: governing_current_policy", messages[1]["content"])
        self.assertIn("Evidence role: historical_reference", messages[1]["content"])
        self.assertIn("NOT GOVERNING CURRENT POLICY", messages[1]["content"])

    @patch("rag.chat")
    def test_refund_dates_are_calculated_deterministically_without_llm(self, mock_chat):
        evidence = [
            hit("REF-2", "| UPI | 1 to 3 business days |", {
                "section_id": "REF-2", "status": "current",
            }),
            hit("KB-00.2", "Frozen date: 20 October 2026.", {
                "section_id": "KB-00.2", "status": "current",
            }),
            hit("KB-00.3", "| 2026-10-19 | Warehouse holiday |", {
                "section_id": "KB-00.3", "status": "current",
            }),
        ]

        result = answer_with_evidence(
            "UPI refund started on 17 October 2026.",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertIn("20 October 2026 and 22 October 2026", result.answer)
        self.assertIn("[REF-2] [KB-00.3]", result.answer)
        self.assertEqual(result.grounding_status, "deterministic_policy_calculation")
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_missing_condition_evidence_abstains_before_llm(self, mock_chat):
        evidence = [
            hit("RET-2", "Beauty return window.", {
                "section_id": "RET-2", "status": "current",
            }),
        ]

        result = answer_with_evidence(
            "I opened the lipstick. Can I return it?",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertIn("cannot confirm", result.answer)
        self.assertEqual(result.grounding_status, "insufficient_required_evidence")
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_damage_report_answer_needs_reporting_citation_not_unasked_remedy(
        self,
        mock_chat,
    ):
        evidence = [
            hit(
                "DWM-1",
                "Report within **48 hours of delivery**, with photos of the "
                "item and the packaging.",
                {
                    "section_id": "DWM-1",
                    "status": "current",
                },
            ),
            hit("DWM-2", "After verification, free replacement or full refund.", {
                "section_id": "DWM-2", "status": "current",
            }),
            hit("RET-6", "No pickup fee applies to damaged items.", {
                "section_id": "RET-6", "status": "current",
            }),
            hit("FAQH-3", "Report within 48 hours with photos.", {
                "section_id": "FAQH-3", "status": "current",
            }),
        ]

        result = answer_with_evidence(
            "मेरा सामान टूटा हुआ आया है, अब क्या करूँ?",
            "hi",
            EvidenceRetriever(evidence),
        )

        self.assertIn("[DWM-1]", result.answer)
        self.assertIn("डिलीवरी के 48 घंटे", result.answer)
        self.assertIn("आइटम", result.answer)
        self.assertIn("पैकेजिंग की फ़ोटो", result.answer)
        self.assertNotIn("रिफ़ंड", result.answer)
        self.assertNotIn("रिप्लेसमेंट", result.answer)
        self.assertNotIn("टिकट", result.answer)
        self.assertEqual(result.grounding_status, "deterministic_policy_response")
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_unclear_damage_reporting_policy_falls_back_without_llm(
        self,
        mock_chat,
    ):
        question = "मेरा सामान टूटा हुआ आया है, अब क्या करूँ?"
        invalid_policies = {
            "packaging-only photos": (
                "Report within 48 hours of delivery, with photos of the packaging."
            ),
            "purchase-anchored window": (
                "Report within 48 hours of purchase, with photos of the item "
                "and the packaging."
            ),
            "conflicting windows": (
                "Report within 24 hours of delivery, with photos of the item "
                "and the packaging. Report within 48 hours of delivery, with "
                "photos of the item and the packaging."
            ),
            "missing packaging requirement": (
                "Report within 48 hours of delivery, with photos of the item."
            ),
            "malformed window": (
                "Report within forty-eight hours of delivery, with photos of "
                "the item and the packaging."
            ),
            "ambiguous time anchor": (
                "Report within 48 hours, with photos of the item and the packaging."
            ),
            "conflicting anchor alternatives": (
                "Report within 48 hours of delivery or purchase, with photos "
                "of the item and the packaging."
            ),
            "photos are optional": (
                "Report within 48 hours of delivery, with photos of the item "
                "and the packaging if available; photos are optional."
            ),
        }

        for case, policy_text in invalid_policies.items():
            with self.subTest(case=case):
                evidence = [
                    hit(
                        "DWM-1",
                        policy_text,
                        {"section_id": "DWM-1", "status": "current"},
                    ),
                    hit(
                        "DWM-2",
                        "After verification, free replacement or full refund.",
                        {"section_id": "DWM-2", "status": "current"},
                    ),
                ]

                result = answer_with_evidence(
                    question,
                    "hi",
                    EvidenceRetriever(evidence),
                )

                self.assertIn("पुष्टि नहीं कर सकता", result.answer)
                self.assertEqual(
                    result.grounding_status,
                    "insufficient_reporting_policy_evidence",
                )
                mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_already_filed_damage_report_skips_reporting_only_response(
        self,
        mock_chat,
    ):
        mock_chat.return_value.message.content = model_response(
            "The policy says a replacement or refund follows verification "
            "[DWM-2]. The reporting window is covered by [DWM-1]."
        )
        evidence = [
            hit(
                "DWM-1",
                "Report within 48 hours of delivery, with photos of the item "
                "and the packaging.",
                {"section_id": "DWM-1", "status": "current"},
            ),
            hit(
                "DWM-2",
                "After verification, free replacement or full refund.",
                {"section_id": "DWM-2", "status": "current"},
            ),
        ]

        result = answer_with_evidence(
            "My damaged item arrived broken. I already filed a complaint. "
            "What should I do now?",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(result.grounding_status, "model_generated")
        self.assertNotIn("Report the issue within", result.answer)
        mock_chat.assert_called_once()

    @patch("rag.chat")
    def test_high_value_damage_requires_ticket_but_does_not_claim_creation(
        self,
        mock_chat,
    ):
        evidence = [
            hit(section_id, f"Policy {section_id}.", {
                "section_id": section_id,
                "status": "current",
            })
            for section_id in ("DWM-1", "DWM-2", "DWM-3", "ESC-1", "ESC-2")
        ]
        evidence.append(hit(
            "products.csv:ELE-0023",
            "Kestrel laptop catalog row.",
            {
                "source_type": "csv",
                "csv_table": "products",
                "record_key": "ELE-0023",
                "row": {
                    "sku": "ELE-0023",
                    "title": "Kestrel Book 14 Laptop",
                    "price_inr": "45999",
                },
            },
        ))

        result = answer_with_evidence(
            "The cracked Kestrel Book 14 Laptop SKU ELE-0023 arrived yesterday.",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(
            result.required_action,
            "create_ticket_then_priority_handoff",
        )
        self.assertEqual(result.action_status, "not_executed")
        self.assertIn("no ticket has been created", result.answer)
        self.assertIn("Priority Resolution team", result.answer)
        self.assertIn("ticket before handoff", result.answer)
        self.assertIn("[DWM-3] [ESC-1] [ESC-2]", result.answer)
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_cod_refund_options_are_derived_from_both_current_policies(
        self,
        mock_chat,
    ):
        evidence = [
            hit(
                "REF-2",
                "| Bank transfer (NEFT) for cash-on-delivery orders | "
                "5 to 7 business days |\n"
                "| ZentroMart Wallet for cash-on-delivery orders | "
                "Instant, within 2 hours |",
                {"section_id": "REF-2", "status": "current"},
            ),
            hit(
                "REF-3",
                "For cash-on-delivery orders, the customer chooses either "
                "ZentroMart Wallet (instant) or a bank transfer. Bank "
                "transfers start only after the bank account details are "
                "verified, then take 5 to 7 business days.",
                {"section_id": "REF-3", "status": "current"},
            ),
        ]

        result = answer_with_evidence(
            "I paid cash on delivery and returned the item. How do I get my money back?",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertIn("ZentroMart Wallet (Instant, within 2 hours)", result.answer)
        self.assertIn("after your bank details are verified", result.answer)
        self.assertIn("5 to 7 business days", result.answer)
        self.assertIn("[REF-3] [REF-2]", result.answer)
        self.assertEqual(result.grounding_status, "deterministic_policy_options")
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_unrecognized_order_requires_ticket_but_does_not_claim_handoff(
        self,
        mock_chat,
    ):
        evidence = [
            hit("ESC-1", "Urgently hand over unauthorized orders.", {
                "section_id": "ESC-1", "status": "current",
            }),
            hit("ESC-2", "Create a ticket before human handoff.", {
                "section_id": "ESC-2", "status": "current",
            }),
        ]

        result = answer_with_evidence(
            "There's an order on my account that I never placed.",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(result.required_action, "create_ticket_then_human_handoff")
        self.assertEqual(result.action_status, "not_executed")
        self.assertIn("No ticket was created", result.answer)
        self.assertIn("no handoff was completed", result.answer)
        self.assertIn("[ESC-1] [ESC-2]", result.answer)
        self.assertEqual(result.grounding_status, "deterministic_policy_workflow")
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_answer_with_unretrieved_citation_falls_back_safely(self, mock_chat):
        mock_chat.return_value.message.content = model_response(
            "Wallet refunds are instant [REF-3]."
        )

        result = answer_with_evidence(
            "How fast are wallet refunds?",
            "en",
            FakeRetriever(),
        )

        self.assertEqual(result.grounding_status, "citation_validation_failed")
        self.assertIn("cannot confirm", result.answer)

    @patch("rag.chat")
    def test_assured_warranty_requires_both_policies_and_product_citation(
        self,
        mock_chat,
    ):
        mock_chat.return_value.message.content = model_response(
            "It has a 12-month warranty [WAR-1]."
        )
        evidence = [
            hit("WAR-1", "Brand warranty.", {
                "section_id": "WAR-1", "status": "current",
            }),
            hit("WAR-2", "Assured adds 12 months.", {
                "section_id": "WAR-2", "status": "current",
            }),
            hit("products.csv:ELE-0008", "Product row.", {
                "source_type": "csv",
                "csv_table": "products",
                "record_key": "ELE-0008",
                "row": {
                    "sku": "ELE-0008",
                    "title": "Voltra PowerCell",
                    "assured": "True",
                },
            }),
        ]

        result = answer_with_evidence(
            "What warranty comes with SKU ELE-0008?",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(result.grounding_status, "citation_validation_failed")
        self.assertIn("cannot confirm", result.answer)

    def test_required_citations_follow_the_question_intent(self):
        evidence = [
            hit("RET-2", "Return window.", {"section_id": "RET-2"}),
            hit("RET-4", "Opened beauty products cannot be returned.", {
                "section_id": "RET-4",
            }),
            hit("RET-6", "Plus members pay no pickup fee.", {
                "section_id": "RET-6",
            }),
            hit("PLS-2", "Plus members receive free reverse pickup.", {
                "section_id": "PLS-2",
            }),
            hit("RET-8", "Festive sale return window is 30 days.", {
                "section_id": "RET-8",
            }),
            hit("WAR-1", "Manufacturer warranty.", {"section_id": "WAR-1"}),
            hit("WAR-2", "Assured coverage.", {"section_id": "WAR-2"}),
            hit("products.csv:ELE-0012", "Assured product.", {
                "source_type": "csv",
                "csv_table": "products",
                "record_key": "ELE-0012",
                "row": {"sku": "ELE-0012", "assured": "True"},
            }),
        ]

        self.assertEqual(
            _required_citations(
                "As a ZentroMart Plus member, will I be charged the pickup fee "
                "for returning shoes?",
                evidence,
            ),
            {"RET-6", "PLS-2"},
        )
        self.assertEqual(
            _required_citations(
                "I opened the Neelam Eau de Parfum. Can I still return it?",
                evidence,
            ),
            {"RET-4"},
        )
        self.assertEqual(
            _required_citations(
                "I bought the kurta in the festive sale. How long can I return it?",
                evidence,
            ),
            {"RET-8"},
        )
        self.assertEqual(
            _required_citations(
                "SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?",
                evidence,
            ),
            {"WAR-2", "products.csv:ELE-0012"},
        )
        self.assertEqual(
            _required_citations(
                "What warranty comes with SKU ELE-0012?",
                evidence,
            ),
            {"WAR-1", "WAR-2", "products.csv:ELE-0012"},
        )
        damage_evidence = [
            hit("DWM-1", "Reporting window.", {"section_id": "DWM-1"}),
            hit("DWM-2", "Resolution.", {"section_id": "DWM-2"}),
            hit("RET-6", "Pickup fee.", {"section_id": "RET-6"}),
        ]
        self.assertEqual(
            _required_citations(
                "मेरा सामान टूटा हुआ आया है, अब क्या करूँ?",
                damage_evidence,
                "48 घंटे के अंदर फ़ोटो के साथ रिपोर्ट करें [DWM-1].",
            ),
            {"DWM-1"},
        )
        self.assertEqual(
            _required_citations(
                "My damaged item arrived. Can I get a refund?",
                damage_evidence,
                "A full refund is available [DWM-2].",
            ),
            {"DWM-1", "DWM-2"},
        )
        self.assertEqual(
            _required_citations(
                "Will I pay a pickup fee for a damaged item?",
                damage_evidence,
                "No pickup fee applies [RET-6].",
            ),
            {"DWM-1", "RET-6"},
        )
        self.assertEqual(
            _required_citations(
                "Is the collection free if the jeans came torn?",
                damage_evidence,
                "The collection is free for this damaged item [RET-6].",
            ),
            {"DWM-1", "RET-6"},
        )

    @patch("rag.chat")
    def test_action_completion_claim_is_replaced_by_transparent_status(
        self,
        mock_chat,
    ):
        mock_chat.return_value.message.content = model_response(
            "Please hold while I connect you to a colleague [ESC-1] [ESC-2]."
        )
        evidence = [
            hit("ESC-1", "Refer urgent cases to a colleague.", {
                "section_id": "ESC-1", "status": "current",
            }),
            hit("ESC-2", "Create a ticket before human handoff.", {
                "section_id": "ESC-2", "status": "current",
            }),
        ]

        result = answer_with_evidence(
            "Can you escalate this to a colleague?",
            "en",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(result.action_status, "not_executed")
        self.assertIn("no handoff was completed", result.answer)
        self.assertEqual(
            result.grounding_status,
            "unexecuted_action_claim_rejected",
        )

    @patch("rag.chat")
    def test_pq0048_deterministic_reporting_answer(self, mock_chat):
        # Use the exact DWM-1 evidence text from records.json
        evidence = [
            hit(
                "DWM-1",
                "Report a damaged, defective, expired, wrong or missing item within **48 hours of delivery**, with photos of the item and the packaging. This applies to every category, including grocery and other non-returnable items.",
                {"section_id": "DWM-1", "status": "current"},
            ),
        ]

        result = answer_with_evidence(
            "मेरा सामान टूटा हुआ आया है, अब क्या करूँ?",
            "hi",
            EvidenceRetriever(evidence),
        )

        self.assertEqual(result.grounding_status, "deterministic_policy_response")
        self.assertIn("[DWM-1]", result.answer)
        self.assertIn("डिलीवरी के 48 घंटे", result.answer)
        self.assertIn("आइटम और पैकेजिंग की फ़ोटो", result.answer)
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_deterministic_return_window_success(self, mock_chat):
        # PQ0024 case: Fashion product + RET-2 policy
        evidence = [
            hit(
                "products.csv:FSH-0026",
                "Fashion product record",
                {
                    "source_type": "csv",
                    "csv_table": "products",
                    "record_key": "FSH-0026",
                    "row": {"sku": "FSH-0026", "title": "Fashion Item", "return_category": "fashion"},
                },
            ),
            hit(
                "RET-2",
                "Return Windows:\nfashion: 15 days\nbeauty: 7 days\nelectronics: 30 days",
                {"section_id": "RET-2", "status": "current"},
            ),
        ]
        result = answer_with_evidence("How long can I return the fashion item FSH-0026?", "en", EvidenceRetriever(evidence))
        self.assertIn("15 days", result.answer)
        self.assertIn("[RET-2]", result.answer)
        self.assertIn("[products.csv:FSH-0026]", result.answer)
        self.assertEqual(result.grounding_status, "deterministic_return_window")
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_deterministic_return_window_fail_closed(self, mock_chat):
        mock_chat.return_value.message.content = model_response("General answer [RET-2].")
        
        # Case 1: Missing product record
        evidence_no_prod = [hit("RET-2", "fashion: 15 days", {"section_id": "RET-2", "status": "current"})]
        res1 = answer_with_evidence("How long can I return fashion items?", "en", EvidenceRetriever(evidence_no_prod))
        # Since the model is called and citations are validated, we just check it's not deterministic
        self.assertNotEqual(res1.grounding_status, "deterministic_return_window")

        # Case 2: Missing RET-2
        evidence_no_pol = [hit("products.csv:FSH-0026", "txt", {"source_type": "csv", "csv_table": "products", "record_key": "FSH-0026", "row": {"return_category": "fashion"}})]
        res2 = answer_with_evidence("How long can I return FSH-0026?", "en", EvidenceRetriever(evidence_no_pol))
        self.assertNotEqual(res2.grounding_status, "deterministic_return_window")

        # Case 3: Outdated RET-2
        evidence_outdated = [
            hit("products.csv:FSH-0026", "txt", {"source_type": "csv", "csv_table": "products", "record_key": "FSH-0026", "row": {"return_category": "fashion"}}),
            hit("RET-2", "fashion: 15 days", {"section_id": "RET-2", "status": "superseded"}),
        ]
        res3 = answer_with_evidence("How long can I return FSH-0026?", "en", EvidenceRetriever(evidence_outdated))
        self.assertNotEqual(res3.grounding_status, "deterministic_return_window")

    @patch("rag.chat")
    def test_deterministic_serviceability_success(self, mock_chat):
        # PQ0037 case: Kochi pincode serviceable=True
        evidence = [
            hit(
                "pincodes.csv:682024",
                "Kochi pincode record",
                {
                    "source_type": "csv",
                    "csv_table": "pincodes",
                    "record_key": "682024",
                    "row": {"pincode": "682024", "serviceable": "True"},
                },
            ),
        ]
        result = answer_with_evidence("Do you deliver to Kochi (682024)?", "en", EvidenceRetriever(evidence))
        self.assertIn("serviceable", result.answer)
        self.assertIn("[pincodes.csv:682024]", result.answer)
        self.assertEqual(result.grounding_status, "deterministic_serviceability")
        mock_chat.assert_not_called()

    @patch("rag.chat")
    def test_deterministic_serviceability_refusal(self, mock_chat):
        evidence = [
            hit(
                "pincodes.csv:110001",
                "Delhi pincode record",
                {
                    "source_type": "csv",
                    "csv_table": "pincodes",
                    "record_key": "110001",
                    "row": {"pincode": "110001", "serviceable": "False"},
                },
            ),
        ]
        result = answer_with_evidence("Do you deliver to 110001?", "en", EvidenceRetriever(evidence))
        self.assertIn("not serviceable", result.answer)
        self.assertIn("[pincodes.csv:110001]", result.answer)
        self.assertEqual(result.grounding_status, "deterministic_serviceability")

    @patch("rag.chat")
    def test_deterministic_serviceability_fail_closed(self, mock_chat):
        # Missing pincode record
        evidence = [hit("SOME-ID", "Random text", {"section_id": "SOME-ID"})]
        result = answer_with_evidence("Do you deliver to 123456?", "en", EvidenceRetriever(evidence))
        self.assertNotEqual(result.grounding_status, "deterministic_serviceability")

    @patch("rag.chat")
    def test_unrelated_questions_follow_normal_rag_path(self, mock_chat):
        mock_chat.return_value.message.content = model_response("Answer [REF-1].")
        evidence = [hit("REF-1", "General policy", {"section_id": "REF-1", "status": "current"})]
        
        result = answer_with_evidence("What is your general policy?", "en", EvidenceRetriever(evidence))
        
        self.assertEqual(result.answer, "Answer [REF-1].")
        self.assertEqual(result.grounding_status, "model_generated")
        mock_chat.assert_called_once()



if __name__ == "__main__":
    unittest.main()
