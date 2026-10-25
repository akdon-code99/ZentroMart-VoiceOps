import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import llm_client
from run_baseline import run_questions


class BaselineAnswerTests(unittest.TestCase):
    @patch("llm_client.chat")
    def test_uses_system_prompt_and_question_only(self, mock_chat):
        mock_chat.return_value.message.content = "  A helpful answer.  "

        answer = llm_client.baseline_answer(
            "Where is my refund?",
            reply_language="en",
        )

        self.assertEqual(answer, "A helpful answer.")
        messages = mock_chat.call_args.args[0]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("ZentroMart's AI customer-support assistant", messages[0]["content"])
        self.assertIn("Never invent or guess policies", messages[0]["content"])
        self.assertIn("Where is my refund?", messages[1]["content"])
        self.assertNotIn("reference_answer", messages[1]["content"])

    def test_rejects_blank_question_without_calling_model(self):
        with patch("llm_client.chat") as mock_chat:
            with self.assertRaises(ValueError):
                llm_client.baseline_answer("  ")
        mock_chat.assert_not_called()


class BaselineEvaluationTests(unittest.TestCase):
    @patch("run_baseline.baseline_answer", return_value="Generated answer")
    def test_retains_gold_fields_but_does_not_pass_them_to_model(self, mock_answer):
        question = {
            "id": "PQ0002",
            "question": "Question text",
            "language": "hi-en",
            "reply_language": "hi-en",
            "expected_decision": "answer",
            "must_include": "required phrase",
            "must_not_include": "forbidden phrase",
            "reference_answer": "gold answer",
        }

        results = run_questions([question])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["generated_answer"], "Generated answer")
        self.assertEqual(results[0]["expected_decision"], "answer")
        self.assertEqual(results[0]["reference_answer"], "gold answer")
        mock_answer.assert_called_once_with("Question text", reply_language="hi-en")

    @patch(
        "run_baseline.baseline_answer",
        side_effect=llm_client.OllamaCloudError("API request failed"),
    )
    def test_records_request_failure_without_fabricating_answer(self, mock_answer):
        results = run_questions([{"id": "PQ0002", "question": "Question text"}])

        self.assertEqual(results[0]["generated_answer"], "")
        self.assertEqual(results[0]["error"], "API request failed")
        mock_answer.assert_called_once()

    def test_results_can_be_saved_separately_as_csv(self):
        row = {
            "id": "PQ0002",
            "question": "Question text",
            "expected_decision": "answer",
            "generated_answer": "Generated answer",
            "error": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline_results.csv"
            pd.DataFrame([row]).to_csv(output, index=False)
            saved = pd.read_csv(output, dtype=str, keep_default_na=False)

        self.assertEqual(saved.loc[0, "question"], "Question text")
        self.assertEqual(saved.loc[0, "generated_answer"], "Generated answer")
        self.assertEqual(saved.loc[0, "expected_decision"], "answer")


if __name__ == "__main__":
    unittest.main()
