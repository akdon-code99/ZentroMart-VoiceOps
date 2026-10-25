"""Measure RAG evidence retrieval against practice/questions.csv (no LLM calls)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from retriever import EvidenceRetriever
from vector_index import DEFAULT_INDEX_DIR, load_vector_index

_PROJECT_ROOT = Path(__file__).resolve().parent
_QUESTIONS_PATH = _PROJECT_ROOT / "practice" / "questions.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "retrieval_evaluation_latest.csv"


def evidence_key(hit) -> str:
    section_id = hit.metadata.get("section_id")
    if section_id:
        return str(section_id)
    if hit.metadata.get("source_type") == "csv":
        return f"{hit.metadata['csv_table']}.csv:{hit.metadata['record_key']}"
    return hit.evidence_id


def retrieval_metrics(gold_evidence: str, hits: list) -> tuple[float | None, float | None]:
    gold = [part.strip() for part in gold_evidence.split(" ; ") if part.strip()]
    if not gold:
        return None, None
    retrieved = [evidence_key(hit) for hit in hits]
    matches = set(gold).intersection(retrieved)
    recall = len(matches) / len(gold)
    reciprocal_rank = next(
        (
            1 / rank
            for rank, key in enumerate(retrieved, start=1)
            if key in matches
        ),
        0.0,
    )
    return recall, reciprocal_rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        help="enable experimental relevance-aware reranking with this candidate pool",
    )
    parser.add_argument("--authority-adjustment", type=float, default=0.03)
    parser.add_argument("--relevance-window", type=float, default=0.002)
    parser.add_argument("--max-product-rows", type=int, default=2)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.candidate_pool is not None and args.candidate_pool < args.top_k:
        parser.error("--candidate-pool must be at least --top-k")

    questions = pd.read_csv(_QUESTIONS_PATH, dtype=str, keep_default_na=False)
    index = load_vector_index(args.index_dir)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(index.manifest["model_name"])
    retriever = EvidenceRetriever(index, encoder)
    selected = questions if args.limit is None else questions.head(args.limit)
    results = []
    superseded_hits = 0
    for number, row in enumerate(selected.to_dict(orient="records"), start=1):
        hits = retriever.search(
            row["question"],
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            authority_adjustment=args.authority_adjustment,
            relevance_window=args.relevance_window,
            max_product_rows=args.max_product_rows,
        )
        superseded_hits += sum(
            hit.metadata.get("status") == "superseded" for hit in hits
        )
        recall, reciprocal_rank = retrieval_metrics(row["gold_evidence"], hits)
        result = dict(row)
        result["retrieved_evidence"] = json.dumps(
            [
                {
                    "evidence_id": hit.evidence_id,
                    "evidence_key": evidence_key(hit),
                    "score": hit.score,
                    "source_path": hit.metadata.get("source_path"),
                    "section_id": hit.metadata.get("section_id"),
                    "version": hit.metadata.get("version"),
                    "status": hit.metadata.get("status"),
                    "doc_type": hit.metadata.get("doc_type"),
                    "retrieval_role": hit.metadata.get("retrieval_role"),
                }
                for hit in hits
            ],
            ensure_ascii=False,
        )
        result["evidence_recall_at_k"] = "" if recall is None else recall
        result["evidence_mrr_at_k"] = "" if reciprocal_rank is None else reciprocal_rank
        results.append(result)
        print(f"Scored {number}/{len(selected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    scored = [row["evidence_recall_at_k"] for row in results if row["evidence_recall_at_k"] != ""]
    if scored:
        mean_recall = sum(scored) / len(scored)
        mean_mrr = sum(row["evidence_mrr_at_k"] for row in results if row["evidence_mrr_at_k"] != "") / len(scored)
        print(f"Answerable/evidence-scored questions: {len(scored)}")
        print(f"Mean evidence recall@{args.top_k}: {mean_recall:.3f}")
        print(f"Mean evidence MRR@{args.top_k}: {mean_mrr:.3f}")
    print(f"Superseded evidence returned: {superseded_hits}")
    print(f"Saved retrieval-only report to {args.output}; no LLM calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
