"""Standalone script to test the ZentroMart conversation agent with text input."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from conversation_runner import ConversationRunner, ConversationMessage
from conversation_agent import RAGConversationResponder
from retriever import EvidenceRetriever
from vector_index import VectorIndex, load_vector_index, DEFAULT_INDEX_DIR
from sentence_transformers import SentenceTransformer

def main():
    # 1. Setup Retriever dependencies
    index = load_vector_index(DEFAULT_INDEX_DIR)
    encoder = SentenceTransformer(index.manifest["model_name"])
    retriever = EvidenceRetriever(index, encoder)

    # 2. Setup the Agent Responder
    responder = RAGConversationResponder(retriever=retriever)

    # 3. Setup the Conversation Runner
    # We pass None for lookup and ticket functions to avoid any side effects
    runner = ConversationRunner(
        responder=responder,
        lookup_order_fn=None,
        create_ticket_fn=None
    )

    print("ZentroMart Agent Test (Type 'exit' or 'quit' to stop)")
    print("-" * 50)

    conversation_id = "test-session-interactive"
    turn = 1

    while True:
        try:
            query_text = input("Customer message: ").strip()
            if not query_text:
                continue
            if query_text.lower() in {"exit", "quit"}:
                break

            message = ConversationMessage(
                conversation_id=conversation_id,
                turn=turn,
                language="en",
                customer_message=query_text
            )

            result = runner.run_turn(message)
            turn += 1

            print(f"\nAgent Response:\n{result.response}")
            if result.evidence_ids:
                print(f"Sources: {', '.join(result.evidence_ids)}")
            print(f"Decision: {result.decision}")
            print("-" * 50)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
