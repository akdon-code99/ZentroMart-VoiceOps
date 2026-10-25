# Final Project Report: ZentroMart VoiceOps 2026

## 1. Project Overview
The ZentroMart VoiceOps project implements an automated customer support agent designed to handle diverse customer inquiries via chat and voice. The primary objective was to create a system that provides factually grounded answers based on a fictional company's knowledge base (KB), handles multi-turn conversations, performs secure identity verification, and manages escalations through ticket creation.

**Scope of Solution:**
- **Multi-modal Interface**: Integration of Gnani STT/TTS for voice interactions.
- **Grounded RAG**: A retrieval-augmented generation pipeline to ensure answers are derived from policy documents.
- **Deterministic Logic**: Hard-coded paths for high-certainty calculations (refund dates, return windows).
- **Secure Workflows**: Identity-checked order lookups and a structured escalation process.

## 2. System Architecture

### 2.1 Retrieval-Augmented Generation (RAG)
The system utilizes a local NumPy-based vector index built using the intfloat/multilingual-e5-small model. 
- **Retrieval**: Uses a pool-16 relevance-aware ranking system to select the top-k most relevant knowledge chunks.
- **Grounding & Citations**: Every factual claim must be accompanied by a citation (e.g., [RET-2]). The system performs a structural check to ensure every cited ID was actually retrieved.
- **Response Parsing**: A robust JSON parser strips Markdown fences and validates the response schema, preventing "hallucinated" prose.

### 2.2 Deterministic Safeguards
To overcome LLM abstention bias and ensure 100% accuracy for critical data, deterministic paths were implemented:
- **Return Windows**: Logic that links a product's return_category directly to the RET-2 policy.
- **Serviceability**: Direct lookup of delivery status from pincodes.csv.
- **Dated Refunds**: A business-day calculator that accounts for Sundays and holidays.csv entries.

### 2.3 Conversation & Voice Architecture
- **Conversation Runner**: A stateful engine that tracks verification status, topic context, and ticket state.
- **Identity Verification**: Implements a two-attempt proof system (phone suffix or pincode) before releasing private order details.
- **Voice Pipeline**: 
    - **Local Audio**: Successful validation of local microphone recording and speaker playback.
    - **STT**: Integration with Gnani STT was verified as functional in earlier testing.
    - **TTS**: Gnani SSE adapter synthesizing speech using the timbre-v2.5 model.
    - **Batch Processing**: A script to process folders of audio files into transcripts and WAV replies.

## 3. Evaluation and Results

### 3.1 RAG Performance
Evaluation against the 52 practice questions shows significant improvement over the LLM-only baseline:

| Metric | Baseline (LLM-only) | RAG (v5) | Improvement |
| :--- | :---: | :---: | :---: |
| **Non-abstention Rate** | 26.9% | 84.6% | +57.7% |
| **Abstention Rate** | 73.1% | 15.4% | -57.7% |
| **Citation Integrity** | N/A | 84.6% | New Capability |
| **Parser Error Rate** | N/A | 0% | Resolved |

*Note: Non-abstention rate indicates the system provided a response rather than refusing due to lack of policy access; it does not inherently guarantee factual correctness. Citation integrity confirms that cited evidence IDs were successfully retrieved, not their semantic accuracy.*

- **Structural Stability**: The parser fix reduced invalid_structured_response cases from **31 (v4)** to **0 (v5)**.
- **Grounding**: Most responses now include required policy citations, with deterministic paths eliminating abstentions for key product-policy lookups (e.g., PQ0024, PQ0037).
- **Regression**: The offline test suite passed **47/47** tests, verifying the integrity of the retriever and RAG logic.

### 3.2 Retrieval Metrics
The retriever demonstrates strong performance in surfacing necessary policy sections, achieving an average **Recall@K of 91%** and an **MRR (Mean Reciprocal Rank) of 0.64** across the practice dataset.

## 4. Limitations and Future Work

### 4.1 Identified Blockers
- **Gnani TTS Connectivity**: Live TTS requests consistently returned **HTTP 403 Forbidden (Error 1010)**. While the adapter logic and local audio playback are verified, the cause of this API rejection was not conclusively established.
- **End-to-End Voice Validation**: Because the TTS component remained blocked, the complete end-to-end voice-to-voice workflow was not fully validated.

### 4.2 Future Improvements
- **Semantic Validation**: Moving beyond structural citation checks to automated semantic verification.
- **Expanded Tooling**: Integrating real-time ticket tracking and live agent handoff.

## 5. Reproducibility

### 5.1 Environment Setup
`powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
`

### 5.2 Execution Commands
- **Index Build**: python build_rag_index.py
- **RAG Evaluation**: python run_rag.py --output outputs/rag_results.csv
- **Offline Tests**: .venv\Scripts\python.exe -m pytest tests/test_rag.py
- **Batch Voice**: python scripts/run_voice_batch.py --input-dir .\audio\incoming --output-dir .\outputs\voice_batch

---
**Date**: 25 October 2026
**Project**: ZentroMart VoiceOps Capstone
