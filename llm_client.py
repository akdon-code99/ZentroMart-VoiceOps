"""Minimal Ollama Cloud chat client for the VoiceOps baseline."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import load_dotenv
import httpx
from ollama import ChatResponse, Client, RequestError, ResponseError

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

_BASELINE_SYSTEM_PROMPT = """You are ZentroMart's AI customer-support assistant.
Begin every standalone reply by briefly introducing yourself as ZentroMart's AI
customer-support assistant, using the requested reply language. Be polite,
clear, and concise. You do not have
access to company policies, knowledge-base documents, product records, customer
accounts, orders, or support tools. Never invent or guess policies, eligibility,
refund timelines, delivery dates, compensation amounts, or account/order facts.
If the question requires information you do not have, say honestly that you
cannot confirm it and offer to connect the customer with a human colleague.
Answer only from information in the customer's message or reliable general
knowledge that does not imply a ZentroMart-specific policy or account fact.
Do not claim that you checked an order, created a ticket, or took an action."""


class OllamaCloudError(RuntimeError):
    """A safe, user-facing Ollama Cloud request error."""

    def __init__(self, message: str, *, category: str = "request") -> None:
        super().__init__(message)
        self.category = category


def _configuration() -> tuple[str, str, str]:
    host = os.getenv("OLLAMA_HOST", "").strip().rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "").strip()
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()

    if host != "https://ollama.com":
        raise RuntimeError("OLLAMA_HOST must be https://ollama.com for Ollama Cloud.")
    if not model:
        raise RuntimeError("Set OLLAMA_MODEL in the project .env file.")
    if not api_key:
        raise RuntimeError("Set OLLAMA_API_KEY in the project .env file.")
    return host, model, api_key


def validate_configuration() -> tuple[str, str]:
    """Validate required settings without returning or displaying the API key."""
    host, model, _ = _configuration()
    return host, model


def chat(messages: Sequence[Mapping[str, str]]) -> ChatResponse:
    """Send chat messages to the configured Ollama Cloud model."""
    host, model, api_key = _configuration()
    client = Client(
        host=host,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        return client.chat(
            model=model,
            messages=list(messages),
            stream=False,
        )
    except ResponseError as error:
        if error.status_code in (401, 403):
            message = "Ollama Cloud rejected the API key. Check OLLAMA_API_KEY."
            category = "authentication"
        elif error.status_code == 404:
            message = "The configured Ollama Cloud model is unavailable or not accessible."
            category = "model_unavailable"
        elif error.status_code == 429:
            message = "Ollama Cloud rate limit reached. Wait and try again later."
            category = "rate_limit"
        else:
            message = f"Ollama Cloud returned an API error (HTTP {error.status_code})."
            category = "api"
        raise OllamaCloudError(message, category=category) from None
    except (ConnectionError, httpx.RequestError, RequestError):
        raise OllamaCloudError(
            "Could not reach Ollama Cloud. Check your network connection and try again.",
            category="network",
        ) from None


def baseline_answer(question: str, reply_language: str | None = None) -> str:
    """Answer one standalone question without retrieval, tools, or memory."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    user_content = question
    if reply_language:
        user_content = (
            f"Reply in the customer's requested language ({reply_language}).\n\n"
            f"Customer question: {question}"
        )

    response = chat(
        [
            {"role": "system", "content": _BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    return response.message.content.strip()
