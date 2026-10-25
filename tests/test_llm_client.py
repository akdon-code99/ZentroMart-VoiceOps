import unittest
from unittest.mock import patch

import llm_client


class OllamaClientTests(unittest.TestCase):
    @patch("llm_client._configuration")
    @patch("llm_client.Client")
    def test_chat_uses_bearer_key_from_configuration(self, mock_client_class, mock_config):
        fake_key = "unit-test-key-not-a-credential"
        mock_config.return_value = ("https://ollama.com", "test-model", fake_key)
        mock_client = mock_client_class.return_value
        mock_client.chat.return_value = object()

        response = llm_client.chat([{"role": "user", "content": "Test"}])

        self.assertIs(response, mock_client.chat.return_value)
        mock_client_class.assert_called_once_with(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {fake_key}"},
        )
        mock_client.chat.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Test"}],
            stream=False,
        )

    def test_cloud_error_keeps_a_safe_category(self):
        error = llm_client.OllamaCloudError(
            "Ollama Cloud rejected the API key. Check OLLAMA_API_KEY.",
            category="authentication",
        )

        self.assertEqual(error.category, "authentication")
        self.assertNotIn("unit-test-key", str(error))


if __name__ == "__main__":
    unittest.main()
