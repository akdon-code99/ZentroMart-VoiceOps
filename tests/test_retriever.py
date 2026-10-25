import unittest

import numpy as np

from knowledge import EvidenceRecord
from evaluate_retrieval import evidence_key, retrieval_metrics
from retriever import EvidenceRetriever, RetrievalHit
from vector_index import VectorIndex


class FakeQueryEncoder:
    def __init__(self):
        self.texts = []

    def encode(self, texts, **kwargs):
        self.texts.extend(texts)
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def make_record(evidence_id, text, metadata):
    return EvidenceRecord(evidence_id, text, text, metadata)


class RetrieverTests(unittest.TestCase):
    def _retriever(self, records):
        return EvidenceRetriever(
            VectorIndex(
                records,
                np.tile(
                    np.asarray([[1.0, 0.0]], dtype=np.float32),
                    (len(records), 1),
                ),
                {"vector_dimensions": 2},
            ),
            FakeQueryEncoder(),
        )

    def test_excludes_preambles_only_when_the_document_has_sections(self):
        records = [
            make_record("policy-preamble", "General policy introduction.", {
                "source_path": "kb/pol_returns_v4.md",
                "status": "current",
            }),
            make_record("RET-2", "Return window policy.", {
                "source_path": "kb/pol_returns_v4.md",
                "status": "current",
                "section_id": "RET-2",
            }),
            make_record("standalone", "Unsectioned guidance.", {
                "source_path": "kb/standalone.md",
                "status": "current",
            }),
            make_record("products.csv:ELE-0001", "Product row.", {
                "source_type": "csv",
                "csv_table": "products",
                "record_key": "ELE-0001",
            }),
        ]

        self.assertEqual(self._retriever(records)._eligible_indices(), [1, 2, 3])

    def test_specific_refund_method_routes_current_timeline_and_excludes_faq(self):
        records = [
            make_record("FAQH-2", "COD bank transfers take 5 to 7 days.", {
                "doc_type": "faq", "status": "current", "section_id": "FAQH-2",
            }),
            make_record("REF-2", "Net banking refund: 3 to 5 business days.", {
                "doc_type": "policy", "status": "current", "section_id": "REF-2",
            }),
            make_record("KB-00.2", "Today's date is 20 October 2026.", {
                "doc_type": "meta", "status": "current", "section_id": "KB-00.2",
            }),
            make_record("KB-00.3", "Business day calendar and holidays.", {
                "doc_type": "meta", "status": "current", "section_id": "KB-00.3",
            }),
        ]

        hits = self._retriever(records).search(
            "Net banking refund started on 17 October 2026.",
            top_k=4,
            candidate_pool=4,
        )

        ids = {hit.metadata.get("section_id") for hit in hits}
        self.assertTrue({"REF-2", "KB-00.2", "KB-00.3"}.issubset(ids))
        self.assertNotIn("FAQH-2", ids)
        self.assertEqual(len(hits), 3)

    def test_cod_refund_routes_cod_policy_instead_of_faq(self):
        records = [
            make_record("FAQH-2", "COD bank transfer takes 5 to 7 days.", {
                "doc_type": "faq", "status": "current", "section_id": "FAQH-2",
            }),
            make_record("REF-2", "COD bank transfer: 5 to 7 business days.", {
                "doc_type": "policy", "status": "current", "section_id": "REF-2",
            }),
            make_record("REF-3", "For COD, bank details are verified first.", {
                "doc_type": "policy", "status": "current", "section_id": "REF-3",
            }),
        ]

        hits = self._retriever(records).search(
            "I paid cash on delivery and returned it. How do I get my money back?",
            top_k=3,
        )

        ids = {hit.metadata.get("section_id") for hit in hits}
        self.assertEqual(ids, {"REF-2", "REF-3"})

    def test_warranty_question_routes_both_warranty_policies_and_exact_product(self):
        records = [
            make_record("WAR-1", "Brand warranty details.", {
                "doc_type": "policy", "status": "current", "section_id": "WAR-1",
            }),
            make_record("WAR-2", "Assured coverage is additional.", {
                "doc_type": "policy", "status": "current", "section_id": "WAR-2",
            }),
            make_record("products.csv:ELE-0008", "Power bank row.", {
                "source_type": "csv", "csv_table": "products",
                "record_key": "ELE-0008",
                "row": {
                    "sku": "ELE-0008",
                    "title": "Voltra PowerCell",
                    "assured": "True",
                },
            }),
        ]

        hits = self._retriever(records).search(
            "What warranty comes with SKU ELE-0008?",
            top_k=3,
        )

        self.assertEqual(
            {hit.evidence_id for hit in hits},
            {"WAR-1", "WAR-2", "products.csv:ELE-0008"},
        )

    def test_opened_beauty_return_routes_condition_and_category_sections(self):
        records = [
            make_record("RET-2", "Beauty return window.", {
                "doc_type": "policy", "status": "current", "section_id": "RET-2",
            }),
            make_record("RET-4", "Beauty must be unopened and sealed.", {
                "doc_type": "policy", "status": "current", "section_id": "RET-4",
            }),
            make_record("products.csv:BTY-0052", "Gulmohar Matte Lipstick.", {
                "source_type": "csv", "csv_table": "products",
                "record_key": "BTY-0052",
                "row": {
                    "sku": "BTY-0052",
                    "title": "Gulmohar Matte Lipstick",
                    "return_category": "beauty_sealed",
                },
            }),
        ]

        hits = self._retriever(records).search(
            "I opened the Gulmohar Matte Lipstick; can I return it?",
            top_k=3,
        )

        self.assertEqual(
            {hit.evidence_id for hit in hits},
            {"RET-2", "RET-4", "products.csv:BTY-0052"},
        )

    def test_high_value_damage_routes_escalation_and_resolution_policy(self):
        sections = ("DWM-1", "DWM-2", "DWM-3", "RET-6", "ESC-1", "ESC-2")
        records = [
            make_record(section_id, f"Current policy {section_id}.", {
                "doc_type": "policy" if not section_id.startswith("ESC-") else "agent_guide",
                "status": "current",
                "section_id": section_id,
            })
            for section_id in sections
        ]
        records.append(make_record("products.csv:ELE-0023", "Laptop row.", {
            "source_type": "csv", "csv_table": "products",
            "record_key": "ELE-0023",
            "row": {
                "sku": "ELE-0023",
                "title": "Kestrel Book 14 Laptop",
                "price_inr": "45999",
            },
        }))

        hits = self._retriever(records).search(
            "The cracked Kestrel Book 14 Laptop SKU ELE-0023 arrived yesterday.",
            top_k=7,
        )

        self.assertTrue(
            set(sections).issubset({hit.metadata.get("section_id") for hit in hits})
        )
        self.assertIn("products.csv:ELE-0023", {hit.evidence_id for hit in hits})
        self.assertLessEqual(len(hits), 7)

    def test_damage_only_routes_pickup_policy_when_fee_is_asked(self):
        retriever = self._retriever([])

        self.assertEqual(
            set(retriever._required_section_ids(
                "मेरा सामान टूटा हुआ आया है, अब क्या करूँ?",
                [],
            )),
            {"DWM-1", "DWM-2"},
        )
        self.assertEqual(
            set(retriever._required_section_ids(
                "Will I pay a pickup fee for a damaged item?",
                [],
            )),
            {"DWM-1", "DWM-2", "RET-6"},
        )
        self.assertEqual(
            set(retriever._required_section_ids(
                "Is the collection free if the jeans came torn?",
                [],
            )),
            {"DWM-1", "DWM-2", "RET-6"},
        )

    def test_unrecognized_order_routes_escalation_guide(self):
        records = [
            make_record("ESC-1", "Hand over orders the customer did not place.", {
                "doc_type": "agent_guide",
                "status": "current",
                "section_id": "ESC-1",
            }),
            make_record("ESC-2", "Create a ticket before human handoff.", {
                "doc_type": "agent_guide",
                "status": "current",
                "section_id": "ESC-2",
            }),
            make_record("FAQ-1", "Order tracking FAQ.", {
                "doc_type": "faq", "status": "current", "section_id": "FAQ-1",
            }),
        ]

        hits = self._retriever(records).search(
            "I never ordered this item.",
            top_k=2,
        )

        self.assertEqual(
            {hit.metadata.get("section_id") for hit in hits},
            {"ESC-1", "ESC-2"},
        )

    def test_excludes_superseded_and_orders_policy_above_faq(self):
        records = [
            make_record("FAQ-1", "FAQ answer", {
                "doc_type": "faq", "status": "current", "section_id": "FAQ-1",
            }),
            make_record("RET-V3", "Old return rule", {
                "doc_type": "policy", "status": "superseded", "section_id": "RETv3-2",
            }),
            make_record("REF-2", "Current refund policy", {
                "doc_type": "policy", "status": "current", "section_id": "REF-2",
            }),
            make_record("KB-00.1", "Policy precedence", {
                "doc_id": "KB-00", "doc_type": "meta", "status": "current",
                "section_id": "KB-00.1",
            }),
        ]
        vectors = np.asarray(
            [[0.8, 0.6], [1.0, 0.0], [0.9, 0.4], [0.7, 0.7]],
            dtype=np.float32,
        )
        encoder = FakeQueryEncoder()
        retriever = EvidenceRetriever(
            VectorIndex(records, vectors, {"vector_dimensions": 2}),
            encoder,
        )

        hits = retriever.search("wallet refund timeline", top_k=3)

        ids = [hit.evidence_id for hit in hits]
        self.assertNotIn("RET-V3", ids)
        self.assertLess(ids.index("REF-2"), ids.index("FAQ-1"))
        self.assertEqual(encoder.texts, ["query: wallet refund timeline"])

    def test_exact_sku_record_is_added_even_if_not_in_vector_top_k(self):
        records = [
            make_record("REF-2", "refund", {
                "doc_type": "policy", "status": "current", "section_id": "REF-2",
            }),
            make_record("products.csv:ELE-0012", "catalog row", {
                "source_type": "csv", "csv_table": "products",
                "record_key": "ELE-0012", "source_path": "data/products.csv",
            }),
        ]
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        retriever = EvidenceRetriever(
            VectorIndex(records, vectors, {"vector_dimensions": 2}),
            FakeQueryEncoder(),
        )

        hits = retriever.search("Is SKU ELE-0012 Assured?", top_k=1)

        self.assertIn("products.csv:ELE-0012", [hit.evidence_id for hit in hits])
        self.assertEqual(len(hits), 1)

    def test_exact_sku_rescue_never_exceeds_top_k(self):
        records = [
            make_record(f"products.csv:ELE-{sku:04d}", "catalog row", {
                "source_type": "csv",
                "csv_table": "products",
                "record_key": f"ELE-{sku:04d}",
            })
            for sku in range(1, 4)
        ]
        vectors = np.asarray(
            [[0.0, 1.0], [0.2, 0.8], [1.0, 0.0]],
            dtype=np.float32,
        )
        retriever = EvidenceRetriever(
            VectorIndex(records, vectors, {"vector_dimensions": 2}),
            FakeQueryEncoder(),
        )

        hits = retriever.search(
            "Compare SKU ELE-0001 and SKU ELE-0002",
            top_k=1,
        )

        self.assertLessEqual(len(hits), 1)
        self.assertIn("products.csv:ELE-0001", [hit.evidence_id for hit in hits])

    def test_candidate_pool_reranking_keeps_relevance_and_caps_product_rows(self):
        records = [
            make_record("FAQ-1", "near match", {
                "doc_type": "faq", "status": "current", "section_id": "FAQ-1",
            }),
            make_record("RET-1", "relevant policy", {
                "doc_type": "policy", "status": "current", "section_id": "RET-1",
            }),
            make_record("products.csv:SKU-0001", "product one", {
                "source_type": "csv", "csv_table": "products",
                "record_key": "SKU-0001",
            }),
            make_record("products.csv:SKU-0002", "product two", {
                "source_type": "csv", "csv_table": "products",
                "record_key": "SKU-0002",
            }),
            make_record("products.csv:SKU-0003", "product three", {
                "source_type": "csv", "csv_table": "products",
                "record_key": "SKU-0003",
            }),
            make_record("POL-LOW", "low relevance policy", {
                "doc_type": "policy", "status": "current", "section_id": "POL-LOW",
            }),
        ]
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.9995, 0.0316],
                [0.999, 0.0447],
                [0.9985, 0.0548],
                [0.998, 0.0632],
                [0.8, 0.6],
            ],
            dtype=np.float32,
        )
        retriever = EvidenceRetriever(
            VectorIndex(records, vectors, {"vector_dimensions": 2}),
            FakeQueryEncoder(),
        )

        hits = retriever.search(
            "return question",
            top_k=3,
            candidate_pool=6,
            authority_adjustment=0.001,
            relevance_window=0.002,
            max_product_rows=1,
        )

        ids = [hit.evidence_id for hit in hits]
        self.assertEqual(len(hits), 3)
        self.assertNotIn("POL-LOW", ids)
        self.assertLessEqual(
            sum(hit.metadata.get("source_type") == "csv" for hit in hits),
            1,
        )
        self.assertIn("RET-1", ids)

    def test_candidate_pool_parameters_are_validated(self):
        retriever = EvidenceRetriever(
            VectorIndex(
                [make_record("RET-1", "policy", {
                    "doc_type": "policy", "status": "current",
                })],
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                {"vector_dimensions": 2},
            ),
            FakeQueryEncoder(),
        )

        with self.assertRaisesRegex(ValueError, "candidate_pool"):
            retriever.search("question", top_k=2, candidate_pool=1)
        with self.assertRaisesRegex(ValueError, "authority_adjustment"):
            retriever.search("question", authority_adjustment=0.031)

    def test_historical_policy_is_included_only_when_current_successor_scopes_event(self):
        current_text = (
            "## [RET-3] Change to electronics returns\n"
            "Electronics **delivered before 1 August 2026** keep the earlier "
            "10-day window from version 3."
        )
        historical_text = (
            "## [RETv3-2] Return windows by category\n"
            "| Electronics and accessories | 10 days |"
        )
        records = [
            make_record("RET-3", current_text, {
                "doc_id": "POL-RET",
                "doc_type": "policy",
                "status": "current",
                "effective_from": "2026-08-01",
                "supersedes": "POL-RET-v3",
                "section_id": "RET-3",
            }),
            make_record("RETv3-2", historical_text, {
                "doc_id": "POL-RET-v3",
                "doc_type": "policy",
                "status": "superseded",
                "effective_from": "2026-03-01",
                "superseded_by": "POL-RET",
                "version": "3.0",
                "section_id": "RETv3-2",
            }),
            make_record("products.csv:ELE-0001", "Test Smartwatch, Blue", {
                "source_type": "csv",
                "csv_table": "products",
                "record_key": "ELE-0001",
                "row": {
                    "title": "Test Smartwatch, Blue",
                    "return_category": "electronics",
                },
            }),
            make_record("KB-00.2", "Today's date is 20 October 2026.", {
                "doc_id": "KB-00",
                "doc_type": "meta",
                "status": "current",
                "section_id": "KB-00.2",
            }),
            make_record("FAQ-1", "Current FAQ", {
                "doc_type": "faq", "status": "current", "section_id": "FAQ-1",
            }),
        ]
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.1],
                [0.95, 0.3],
                [0.7, 0.7],
                [0.8, 0.6],
            ],
            dtype=np.float32,
        )
        retriever = EvidenceRetriever(
            VectorIndex(records, vectors, {"vector_dimensions": 2}),
            FakeQueryEncoder(),
        )

        undated_hits = retriever.search(
            "What is the return window for Test Smartwatch?",
            top_k=4,
        )
        historical_hits = retriever.search(
            "Test Smartwatch was delivered on 20 July. What was the return window?",
            top_k=4,
        )

        self.assertNotIn("RETv3-2", [hit.evidence_id for hit in undated_hits])
        self.assertEqual(
            [hit.evidence_id for hit in historical_hits[:2]],
            ["RET-3", "RETv3-2"],
        )
        self.assertEqual(
            historical_hits[0].metadata["retrieval_role"],
            "governing_current_policy",
        )
        self.assertEqual(
            historical_hits[1].metadata["retrieval_role"],
            "historical_reference",
        )
        self.assertIn("NOT GOVERNING CURRENT POLICY", historical_hits[1].text)
        self.assertLessEqual(len(historical_hits), 4)

    def test_historical_policy_is_not_used_for_current_or_unmatched_events(self):
        current_text = (
            "## [RET-3] Change to electronics returns\n"
            "Electronics **delivered before 1 August 2026** keep the earlier "
            "10-day window from version 3."
        )
        records = [
            make_record("RET-3", current_text, {
                "doc_id": "POL-RET",
                "doc_type": "policy",
                "status": "current",
                "effective_from": "2026-08-01",
                "supersedes": "POL-RET-v3",
                "section_id": "RET-3",
            }),
            make_record("RETv3-2", "| Electronics and accessories | 10 days |", {
                "doc_id": "POL-RET-v3",
                "doc_type": "policy",
                "status": "superseded",
                "effective_from": "2026-03-01",
                "superseded_by": "POL-RET",
                "version": "3.0",
                "section_id": "RETv3-2",
            }),
            make_record("products.csv:ELE-0001", "Test Smartwatch, Blue", {
                "source_type": "csv",
                "csv_table": "products",
                "record_key": "ELE-0001",
                "row": {
                    "title": "Test Smartwatch, Blue",
                    "return_category": "electronics",
                },
            }),
            make_record("KB-00.2", "Today's date is 20 October 2026.", {
                "doc_id": "KB-00",
                "doc_type": "meta",
                "status": "current",
                "section_id": "KB-00.2",
            }),
        ]
        retriever = EvidenceRetriever(
            VectorIndex(
                records,
                np.asarray(
                    [[1.0, 0.0], [0.99, 0.1], [0.95, 0.3], [0.7, 0.7]],
                    dtype=np.float32,
                ),
                {"vector_dimensions": 2},
            ),
            FakeQueryEncoder(),
        )

        hits = retriever.search(
            "Test Smartwatch was delivered on 20 October 2026.",
            top_k=4,
        )

        self.assertNotIn("RETv3-2", [hit.evidence_id for hit in hits])

    def test_supplied_precedence_section_is_available_for_answer_prompt(self):
        record = make_record("KB-00.1", "## [KB-00.1] Which document wins", {
            "doc_id": "KB-00",
            "doc_type": "meta",
            "status": "current",
            "section_id": "KB-00.1",
        })
        retriever = EvidenceRetriever(
            VectorIndex([record], np.asarray([[1.0, 0.0]]), {}),
            FakeQueryEncoder(),
        )

        self.assertIn("Which document wins", retriever.authority_guidance())

    def test_gold_evidence_metrics_support_sections_and_catalog_rows(self):
        hits = [
            RetrievalHit("faq", "", "", {"section_id": "FAQ-1"}, 0.9),
            RetrievalHit(
                "products.csv:ELE-0012",
                "",
                "",
                {
                    "source_type": "csv",
                    "csv_table": "products",
                    "record_key": "ELE-0012",
                },
                0.8,
            ),
            RetrievalHit("WAR-2", "", "", {"section_id": "WAR-2"}, 0.7),
        ]

        self.assertEqual(evidence_key(hits[1]), "products.csv:ELE-0012")
        recall, reciprocal_rank = retrieval_metrics(
            "WAR-2 ; products.csv:ELE-0012",
            hits,
        )

        self.assertEqual(recall, 1.0)
        self.assertEqual(reciprocal_rank, 0.5)
        self.assertEqual(retrieval_metrics("", hits), (None, None))


if __name__ == "__main__":
    unittest.main()
