import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import llm_client
import run_rag
from retriever import RetrievalHit


class RunRAGTests(unittest.TestCase):
    def setUp(self):
        self.hit = RetrievalHit(
            evidence_id="kb/pol_refunds.md::REF-2::chunk",
            text="Wallet refunds are instant.",
            source_text="## [REF-2] Refund timelines",
            metadata={
                "source_path": "kb/pol_refunds.md",
                "doc_id": "POL-REF",
                "section_id": "REF-2",
                "version": "2.1",
                "status": "current",
                "doc_type": "policy",
                "retrieval_role": "governing_current_policy",
            },
            score=0.91,
        )

    @patch("run_rag.answer_with_evidence")
    def test_question_id_filter_runs_only_requested_question(self, mock_answer):
        mock_answer.return_value = SimpleNamespace(
            answer="The reporting window is 48 hours [DWM-1].",
            evidence=[self.hit],
            required_action="",
            action_status="not_required",
            grounding_status="model_generated",
        )
        questions = [
            {"id": "PQ0047", "question": "An unrelated question"},
            {"id": "PQ0048", "question": "How soon should I report damage?"},
            {"id": "PQ0049", "question": "Another unrelated question"},
        ]
        selected = run_rag._select_questions(questions, ["PQ0048"])
        retriever = object()

        results = run_rag.run_questions(selected, retriever)

        self.assertEqual([row["id"] for row in results], ["PQ0048"])
        mock_answer.assert_called_once_with(
            "How soon should I report damage?",
            "",
            retriever,
        )

    def test_question_id_filter_rejects_unknown_ids(self):
        with self.assertRaisesRegex(ValueError, "PQ9999"):
            run_rag._select_questions(
                [{"id": "PQ0048", "question": "Question"}],
                ["PQ9999"],
            )

    @patch("run_rag.answer_with_evidence")
    def test_records_answer_and_evidence_without_passing_evaluation_labels(
        self,
        mock_answer,
    ):
        mock_answer.return_value = SimpleNamespace(
            answer="Wallet refunds are instant [REF-2].",
            evidence=[self.hit],
            required_action="",
            action_status="not_required",
            grounding_status="model_generated",
        )
        question = {
            "id": "PQ0003",
            "question": "How fast are wallet refunds?",
            "reply_language": "en",
            "reference_answer": "Instant, within two hours.",
            "gold_evidence": "REF-2",
        }
        retriever = object()

        results = run_rag.run_questions([question], retriever)

        mock_answer.assert_called_once_with(question["question"], "en", retriever)
        self.assertEqual(results[0]["run_status"], "success")
        self.assertEqual(results[0]["action_status"], "not_required")
        self.assertEqual(results[0]["grounding_status"], "model_generated")
        evidence = json.loads(results[0]["retrieved_evidence"])
        self.assertEqual(evidence[0]["evidence_key"], "REF-2")
        self.assertEqual(evidence[0]["metadata"]["version"], "2.1")
        self.assertNotIn("reference_answer", mock_answer.call_args.args)

    @patch("run_rag.answer_with_evidence")
    def test_authentication_failure_stops_following_requests(self, mock_answer):
        mock_answer.side_effect = llm_client.OllamaCloudError(
            "Ollama Cloud rejected the API key. Check OLLAMA_API_KEY.",
            category="authentication",
        )
        questions = [
            {"id": "PQ0002", "question": "Question 1"},
            {"id": "PQ0003", "question": "Question 2"},
        ]

        results = run_rag.run_questions(questions, object())

        mock_answer.assert_called_once()
        self.assertEqual(results[0]["run_status"], "error")
        self.assertEqual(
            results[0]["error"],
            "Ollama Cloud rejected the API key. Check OLLAMA_API_KEY.",
        )
        self.assertEqual(results[0]["generated_answer"], "")
        self.assertEqual(results[1]["run_status"], "not_attempted")
        self.assertEqual(results[1]["generated_answer"], "")

    def test_csv_contains_evidence_metadata_and_does_not_overwrite_protected_file(self):
        question = {"id": "PQ0003", "question": "How fast?"}
        result = {
            **question,
            "generated_answer": "Instant [REF-2].",
            "retrieved_evidence": run_rag._serialize_evidence([self.hit]),
            "required_action": "",
            "action_status": "not_required",
            "grounding_status": "model_generated",
            "error": "",
            "run_status": "success",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rag_results.csv"
            run_rag.save_results([result], list(question), output)
            saved = pd.read_csv(output, dtype=str, keep_default_na=False)
            evidence = json.loads(saved.loc[0, "retrieved_evidence"])
            self.assertEqual(saved.loc[0, "run_status"], "success")
            self.assertEqual(saved.loc[0, "action_status"], "not_required")
            self.assertEqual(evidence[0]["section_id"], "REF-2")
            self.assertEqual(
                evidence[0]["metadata"]["retrieval_role"],
                "governing_current_policy",
            )

            with self.assertRaisesRegex(ValueError, "existing output"):
                run_rag.save_results([result], list(question), output)

            protected = Path(directory) / "baseline_results.csv"
            protected.write_text("preserve this baseline", encoding="utf-8")
            with patch("run_rag._PROTECTED_OUTPUTS", {protected}):
                with self.assertRaisesRegex(ValueError, "protected"):
                    run_rag.save_results([result], list(question), protected)
            self.assertEqual(
                protected.read_text(encoding="utf-8"),
                "preserve this baseline",
            )

    def test_product_evidence_key_matches_retrieval_evaluation(self):
        product_hit = RetrievalHit(
            evidence_id="products.csv:ELE-0012",
            text="Product row",
            source_text="Product row",
            metadata={
                "source_type": "csv",
                "csv_table": "products",
                "record_key": "ELE-0012",
            },
            score=0.8,
        )

        self.assertEqual(
            run_rag.evidence_key(product_hit),
            "products.csv:ELE-0012",
        )


if __name__ == "__main__":
    unittest.main()
