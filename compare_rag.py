"""Create a baseline/RAG comparison and manual answer-review CSV."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from vector_index import DEFAULT_INDEX_DIR, load_vector_index

_PROJECT_ROOT = Path(__file__).resolve().parent
_BASELINE_PATH = _PROJECT_ROOT / "outputs" / "baseline_results.csv"
_RAG_PATH = _PROJECT_ROOT / "outputs" / "rag_results.csv"
_RETRIEVAL_PATH = _PROJECT_ROOT / "outputs" / "retrieval_evaluation_historical.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "baseline_vs_rag.csv"
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_QUESTION_FIELDS = (
    "language",
    "question",
    "intent",
    "expected_decision",
    "acceptable_decisions",
    "answerable",
    "gold_evidence",
    "hard_negatives",
    "must_include",
    "must_not_include",
    "reference_answer",
    "reply_language",
    "needs_native_review",
    "notes",
)
_REVIEW_FIELDS = (
    "review_supported_by_retrieved_evidence",
    "review_factual_correctness",
    "review_completeness",
    "review_policy_version_correctness",
    "review_abstention_appropriate",
    "review_unsupported_claims",
    "review_notes",
)


def _rows_by_id(frame: pd.DataFrame, label: str) -> dict[str, dict[str, str]]:
    if "id" not in frame.columns:
        raise ValueError(f"{label} CSV has no 'id' column.")
    if frame["id"].eq("").any():
        raise ValueError(f"{label} CSV contains an empty question ID.")
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].tolist()
        raise ValueError(f"{label} CSV contains duplicate question IDs: {duplicates}")
    return {str(row["id"]): row for row in frame.to_dict(orient="records")}


def _retrieved_identities(evidence_json: str) -> set[str]:
    if not evidence_json:
        return set()
    evidence = json.loads(evidence_json)
    if not isinstance(evidence, list):
        raise ValueError("RAG retrieved_evidence must be a JSON list.")

    identities = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Each retrieved evidence entry must be a JSON object.")
        for key in ("evidence_id", "evidence_key", "section_id"):
            value = item.get(key)
            if value:
                normalized = str(value).replace("::", ":")
                identities.update(part for part in normalized.split(":") if part)
                identities.add(str(value))
    return identities


def _citation_review(answer: str, retrieved_evidence: str) -> tuple[str, str]:
    if not answer.strip():
        return "", "not_evaluated"

    citations = []
    for match in _CITATION_PATTERN.finditer(answer):
        for citation in re.split(r"\s*[,;]\s*", match.group(1).strip()):
            citation = citation.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_./:-]*", citation):
                citations.append(citation)
    if not citations:
        return "", "no_citations"

    retrieved = _retrieved_identities(retrieved_evidence)
    cited = "; ".join(dict.fromkeys(citations))
    normalized = [citation.rsplit("::", 1)[-1] for citation in citations]
    if all(
        citation in retrieved or short_id in retrieved
        for citation, short_id in zip(citations, normalized)
    ):
        return cited, "all_cited_ids_retrieved"
    return cited, "contains_unretrieved_citation_ids"


def _retrieval_gap(gold_evidence: str, recall: str) -> str:
    if not gold_evidence.strip():
        return "no_gold_evidence_provided"
    if not recall.strip():
        return "recall_unavailable"
    try:
        score = float(recall)
    except ValueError as error:
        raise ValueError(f"Invalid evidence recall value: {recall}") from error
    if score <= 0:
        return "zero_recall"
    if score < 1:
        return "partial_recall"
    return "full_recall"


def validate_output_path(output: Path, inputs: list[Path]) -> None:
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("--output must be a separate file from all input CSVs.")
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output file: {output}")


def add_evidence_text(
    rag: pd.DataFrame,
    text_by_evidence_id: dict[str, str],
) -> pd.DataFrame:
    enriched = rag.copy()
    if "retrieved_evidence" not in enriched.columns:
        return enriched

    for index, evidence_json in enriched["retrieved_evidence"].items():
        if not evidence_json:
            continue
        evidence = json.loads(evidence_json)
        if not isinstance(evidence, list):
            raise ValueError("RAG retrieved_evidence must be a JSON list.")
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("Each retrieved evidence entry must be a JSON object.")
            if item.get("text"):
                continue
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id not in text_by_evidence_id:
                raise ValueError(
                    f"Retrieved evidence ID is missing from the local index: {evidence_id}"
                )
            item["text"] = text_by_evidence_id[evidence_id]
        enriched.at[index, "retrieved_evidence"] = json.dumps(
            evidence,
            ensure_ascii=False,
        )
    return enriched


def build_comparison(
    baseline: pd.DataFrame,
    rag: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> pd.DataFrame:
    baseline_rows = _rows_by_id(baseline, "Baseline")
    rag_rows = _rows_by_id(rag, "RAG")
    retrieval_rows = _rows_by_id(retrieval, "Retrieval evaluation")
    identifiers = list(dict.fromkeys(
        [*baseline_rows, *rag_rows, *retrieval_rows]
    ))
    comparisons = []

    for question_id in identifiers:
        baseline_row = baseline_rows.get(question_id, {})
        rag_row = rag_rows.get(question_id, {})
        retrieval_row = retrieval_rows.get(question_id, {})
        dataset_row = baseline_row or rag_row or retrieval_row
        baseline_answer = baseline_row.get("generated_answer", "")
        rag_answer = rag_row.get("generated_answer", "")
        retrieved_evidence = rag_row.get("retrieved_evidence", "")
        cited_ids, citation_check = _citation_review(
            rag_answer,
            retrieved_evidence,
        )
        gold_evidence = dataset_row.get("gold_evidence", "")
        recall = retrieval_row.get("evidence_recall_at_k", "")
        baseline_status = (
            "missing_result"
            if not baseline_row
            else ("error" if baseline_row.get("error", "") else "success")
        )
        result = {
            "id": question_id,
            **{field: dataset_row.get(field, "") for field in _QUESTION_FIELDS},
            "baseline_answer": baseline_answer,
            "baseline_error": baseline_row.get("error", ""),
            "baseline_run_status": baseline_status,
            "rag_answer": rag_answer,
            "rag_error": rag_row.get("error", ""),
            "rag_run_status": rag_row.get("run_status", "missing_result"),
            "retrieved_evidence": retrieved_evidence,
            "required_action": rag_row.get("required_action", ""),
            "action_status": rag_row.get("action_status", "not_available"),
            "grounding_status": rag_row.get("grounding_status", ""),
            "evidence_recall_at_k": recall,
            "evidence_mrr_at_k": retrieval_row.get("evidence_mrr_at_k", ""),
            "retrieval_gap_group": _retrieval_gap(gold_evidence, recall),
            "cited_evidence_ids": cited_ids,
            "citation_integrity_check": citation_check,
            "manual_review_status": "pending",
            **{field: "" for field in _REVIEW_FIELDS},
        }
        comparisons.append(result)

    return pd.DataFrame(comparisons)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=_BASELINE_PATH)
    parser.add_argument("--rag", type=Path, default=_RAG_PATH)
    parser.add_argument("--retrieval", type=Path, default=_RETRIEVAL_PATH)
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()

    try:
        validate_output_path(
            args.output,
            [args.baseline, args.rag, args.retrieval],
        )
        baseline = pd.read_csv(args.baseline, dtype=str, keep_default_na=False)
        rag = pd.read_csv(args.rag, dtype=str, keep_default_na=False)
        retrieval = pd.read_csv(args.retrieval, dtype=str, keep_default_na=False)
        index = load_vector_index(args.index_dir)
        text_by_evidence_id = {
            record.evidence_id: record.text
            for record in index.records
        }
        rag = add_evidence_text(rag, text_by_evidence_id)
        comparison = build_comparison(baseline, rag, retrieval)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(f"Saved {len(comparison)} comparison row(s) to {args.output}")
    print("Answer-quality fields are pending manual review; citation checks are structural only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
