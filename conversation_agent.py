"""Conversation responder that reuses the evidence-grounded RAG pipeline."""

from typing import Callable

from conversation_runner import AgentReply, ConversationContext, CustomerInput
from rag import RAGAnswer, answer_with_evidence
from retriever import EvidenceRetriever

RAGAnswerer = Callable[..., RAGAnswer]


class RAGConversationResponder:
    """Adapt the existing RAG answer function to conversation turns."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        rag_answerer: RAGAnswerer = answer_with_evidence,
    ) -> None:
        self._retriever = retriever
        self._rag_answerer = rag_answerer

    def __call__(
        self,
        customer: CustomerInput,
        context: ConversationContext,
    ) -> AgentReply:
        recent_customer_context = [
            entry.content
            for entry in context.history
            if entry.role == "user"
        ][-3:]
        question = customer.customer_message
        if recent_customer_context:
            question = (
                "Earlier customer messages (context only):\n"
                + "\n".join(recent_customer_context)
                + f"\nCurrent customer question:\n{customer.customer_message}"
            )

        result = self._rag_answerer(
            question,
            customer.language,
            self._retriever,
            verified_order_facts=context.verified_order_facts,
        )
        decision = result.decision
        if result.required_action:
            decision = "escalate"
        if decision not in {"answer", "clarify", "escalate", "refuse"}:
            raise ValueError(f"Unsupported conversation decision: {decision}")

        return AgentReply(
            response=result.answer,
            decision=decision,
            required_action=result.required_action,
            evidence_ids=tuple(
                str(hit.metadata.get("section_id") or hit.evidence_id)
                for hit in result.evidence
            ),
            grounding_status=result.grounding_status,
        )
