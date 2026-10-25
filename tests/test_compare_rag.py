import json
import unittest
from pathlib import Path

import pandas as pd

from compare_rag import (
    _citation_review,
    add_evidence_text,
    build_comparison,
    validate_output_path,
)


class CompareRAGTests(unittest.TestCase):
    def setUp(self):
        self.baseline = pd.DataFrame([
            {
                "id": "PQ0002",
                "question": "Does SKU ELE-0012 have Assured?",
                "expected_decision": "answer",
                "acceptable_decisions": "answer",
                "answerable": "True",
                "gold_evidence": "WAR-2 ; products.csv:ELE-0012",
                "reference_answer": "Reference for reviewer only.",
                "generated_answer": "Yes, it is covered.",
                "error": "",
                "must_include": "12 months",
                "must_not_include": "",
            },
            {
                "id": "PQ0004",
                "question": "Can I schedule a midnight delivery?",
                "expected_decision": "abstain",
                "acceptable_decisions": "abstain",
                "answerable": "False",
                "gold_evidence": "",
                "reference_answer": "No scheduling evidence.",
                "generated_answer": "I cannot confirm that.",
                "error": "",
                "must_include": "",
                "must_not_include": "",
            },
        ])
        self.rag = pd.DataFrame([
            {
                "id": "PQ0004",
                "generated_answer": "I cannot confirm this [UNKNOWN].",
                "retrieved_evidence": "[]",
                "error": "",
                "run_status": "success",
            },
            {
                "id": "PQ0002",
                "generated_answer": "It has a 12-month warranty [WAR-2].",
                "retrieved_evidence": (
                    '[{"evidence_id":"chunk-1","evidence_key":"WAR-2",'
                    '"section_id":"WAR-2","metadata":{"version":"1.1"}}]'
                ),
                "required_action": "create_ticket_then_priority_handoff",
                "action_status": "not_executed",
                "grounding_status": "deterministic_policy_workflow",
                "error": "",
                "run_status": "success",
            },
        ])
        self.retrieval = pd.DataFrame([
            {
                "id": "PQ0004",
                "evidence_recall_at_k": "",
                "evidence_mrr_at_k": "",
            },
            {
                "id": "PQ0002",
                "evidence_recall_at_k": "0.5",
                "evidence_mrr_at_k": "0.5",
            },
        ])

    def test_matches_by_id_and_separates_automatic_and_manual_review(self):
        comparison = build_comparison(self.baseline, self.rag, self.retrieval)

        self.assertEqual(comparison["id"].tolist(), ["PQ0002", "PQ0004"])
        pq2 = comparison.set_index("id").loc["PQ0002"]
        self.assertEqual(pq2["rag_answer"], "It has a 12-month warranty [WAR-2].")
        self.assertEqual(pq2["retrieval_gap_group"], "partial_recall")
        self.assertEqual(pq2["citation_integrity_check"], "all_cited_ids_retrieved")
        self.assertEqual(
            pq2["required_action"],
            "create_ticket_then_priority_handoff",
        )
        self.assertEqual(pq2["action_status"], "not_executed")
        self.assertEqual(pq2["manual_review_status"], "pending")
        self.assertEqual(pq2["review_factual_correctness"], "")

        pq4 = comparison.set_index("id").loc["PQ0004"]
        self.assertEqual(pq4["retrieval_gap_group"], "no_gold_evidence_provided")
        self.assertEqual(pq4["answerable"], "False")
        self.assertEqual(
            pq4["citation_integrity_check"],
            "contains_unretrieved_citation_ids",
        )

    def test_duplicate_ids_are_rejected(self):
        duplicate = pd.concat([self.baseline, self.baseline.iloc[[0]]])

        with self.assertRaisesRegex(ValueError, "duplicate question IDs"):
            build_comparison(duplicate, self.rag, self.retrieval)

    def test_comparison_output_must_not_overwrite_an_input(self):
        input_path = Path("baseline_results.csv")

        with self.assertRaisesRegex(ValueError, "separate file"):
            validate_output_path(input_path, [input_path])

        validate_output_path(Path("baseline_vs_rag.csv"), [input_path])

    def test_adds_index_text_for_manual_evidence_review(self):
        enriched = add_evidence_text(
            self.rag,
            {"chunk-1": "Warranty is 12 months."},
        )
        evidence = enriched.loc[
            enriched["id"].eq("PQ0002"),
            "retrieved_evidence",
        ].iloc[0]

        self.assertEqual(json.loads(evidence)[0]["text"], "Warranty is 12 months.")

    def test_citation_check_accepts_retrieved_section_aliases_and_grouped_ids(self):
        evidence = (
            '[{"evidence_id":"kb/pol_returns_v4.md::RET-3::0::abc",'
            '"evidence_key":"RET-3","section_id":"RET-3"},'
            '{"evidence_id":"chunk-ref2","evidence_key":"REF-2",'
            '"section_id":"REF-2"},'
            '{"evidence_id":"products.csv:INR-0067",'
            '"evidence_key":"products.csv:INR-0067"}]'
        )

        _, result = _citation_review(
            "This cites [kb/pol_returns_v4.md::RET-3], [REF-2, RET-3], "
            "and [INR-0067].",
            evidence,
        )

        self.assertEqual(result, "all_cited_ids_retrieved")


if __name__ == "__main__":
    unittest.main()
