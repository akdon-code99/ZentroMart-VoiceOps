"""Evaluate the RAG answer pipeline against practice/questions.csv."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from llm_client import OllamaCloudError, validate_configuration
from rag import answer_with_evidence
from retriever import EvidenceRetriever, RetrievalHit
from vector_index import DEFAULT_INDEX_DIR, load_vector_index

_PROJECT_ROOT = Path(__file__).resolve().parent
_QUESTIONS_PATH = _PROJECT_ROOT / "practice" / "questions.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "rag_results.csv"
_PROTECTED_OUTPUTS = {
    _QUESTIONS_PATH,
    _PROJECT_ROOT / "outputs" / "baseline_results.csv",
    _PROJECT_ROOT / "outputs" / "rag_results.csv",
    _PROJECT_ROOT / "outputs" / "rag_results_smoke.csv",
    _PROJECT_ROOT / "outputs" / "baseline_vs_rag.csv",
    _PROJECT_ROOT / "outputs" / "retrieval_evaluation.csv",
    _PROJECT_ROOT / "outputs" / "retrieval_evaluation_corrected.csv",
    _PROJECT_ROOT / "outputs" / "retrieval_evaluation_experiment.csv",
    _PROJECT_ROOT / "outputs" / "retrieval_evaluation_historical.csv",
    _PROJECT_ROOT / "outputs" / "retrieval_evaluation_latest.csv",
}


def _positive_int(value: str) -> int:
    limit = int(value)
    if limit < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return limit


def _select_questions(
    questions: list[dict[str, str]],
    question_ids: list[str] | None,
) -> list[dict[str, str]]:
    if not question_ids:
        return questions

    requested = set(question_ids)
    selected = [row for row in questions if row.get("id") in requested]
    found = {row.get("id") for row in selected}
    missing = requested - found
    if missing:
        raise ValueError(
            f"Unknown question ID(s): {', '.join(sorted(missing))}"
        )
    return selected


def evidence_key(hit: RetrievalHit) -> str:
    section_id = hit.metadata.get("section_id")
    if section_id:
        return str(section_id)
    if hit.metadata.get("source_type") == "csv":
        return f"{hit.metadata['csv_table']}.csv:{hit.metadata['record_key']}"
    return hit.evidence_id


def _serialize_evidence(hits: list[RetrievalHit]) -> str:
    return json.dumps(
        [
            {
                "evidence_id": hit.evidence_id,
                "evidence_key": evidence_key(hit),
                "text": hit.text,
                "score": hit.score,
                "source_path": hit.metadata.get("source_path"),
                "doc_id": hit.metadata.get("doc_id"),
                "section_id": hit.metadata.get("section_id"),
                "version": hit.metadata.get("version"),
                "status": hit.metadata.get("status"),
                "doc_type": hit.metadata.get("doc_type"),
                "retrieval_role": hit.metadata.get("retrieval_role"),
                "metadata": hit.metadata,
            }
            for hit in hits
        ],
        ensure_ascii=False,
    )


def run_questions(
    questions: list[dict[str, str]],
    retriever: EvidenceRetriever,
    limit: int | None = None,
) -> list[dict[str, str]]:
    selected = questions if limit is None else questions[:limit]
    results = []
    authentication_failed = False

    for number, row in enumerate(selected, start=1):
        result = dict(row)
        result["generated_answer"] = ""
        result["retrieved_evidence"] = "[]"
        result["required_action"] = ""
        result["action_status"] = "not_required"
        result["grounding_status"] = ""
        result["error"] = ""
        result["run_status"] = "not_attempted" if authentication_failed else "pending"

        if authentication_failed:
            results.append(result)
            print(f"Skipped {number}/{len(selected)} after authentication failure")
            continue

        try:
            output = answer_with_evidence(
                row["question"],
                row.get("reply_language") or row.get("language", ""),
                retriever,
            )
            result["generated_answer"] = output.answer
            result["retrieved_evidence"] = _serialize_evidence(output.evidence)
            result["required_action"] = output.required_action
            result["action_status"] = output.action_status
            result["grounding_status"] = output.grounding_status
            result["run_status"] = "success"
        except (OllamaCloudError, RuntimeError, ValueError, KeyError) as error:
            result["error"] = str(error)
            result["run_status"] = "error"
            authentication_failed = (
                isinstance(error, OllamaCloudError)
                and error.category == "authentication"
            )
            print(
                f"Question {row.get('id', number)} failed: {error}",
                file=sys.stderr,
            )

        results.append(result)
        print(f"Processed {number}/{len(selected)}")

    return results


def save_results(
    results: list[dict[str, str]],
    question_columns: list[str],
    output_path: Path,
) -> None:
    resolved_output = output_path.resolve()
    protected_paths = {path.resolve() for path in _PROTECTED_OUTPUTS}
    if resolved_output in protected_paths:
        raise ValueError(
            f"Refusing to overwrite protected input or evaluation file: {output_path}"
        )
    if resolved_output.exists():
        raise ValueError(f"Refusing to overwrite existing output file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        results,
        columns=[
            *question_columns,
            "generated_answer",
            "retrieved_evidence",
            "required_action",
            "action_status",
            "grounding_status",
            "error",
            "run_status",
        ],
    ).to_csv(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--question-ids",
        nargs="+",
        help="Run only these exact practice-question IDs.",
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH)
    args = parser.parse_args()
    if args.limit is not None and args.question_ids:
        parser.error("--limit cannot be combined with --question-ids")

    try:
        validate_configuration()
        questions = pd.read_csv(_QUESTIONS_PATH, dtype=str, keep_default_na=False)
        question_columns = list(questions.columns)
        questions = _select_questions(
            questions.to_dict(orient="records"),
            args.question_ids,
        )
        index = load_vector_index(args.index_dir)
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(index.manifest["model_name"])
        retriever = EvidenceRetriever(index, encoder)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"RAG setup failed: {error}", file=sys.stderr)
        return 2

    results = run_questions(
        questions,
        retriever,
        limit=args.limit,
    )
    try:
        save_results(results, question_columns, args.output)
    except (OSError, ValueError) as error:
        print(f"Could not save RAG results: {error}", file=sys.stderr)
        return 2

    print(f"Saved {len(results)} result(s) to {args.output}")
    failures = sum(row["run_status"] == "error" for row in results)
    not_attempted = sum(row["run_status"] == "not_attempted" for row in results)
    if failures:
        print(f"{failures} request(s) failed; see the error column.")
    if not_attempted:
        print(f"{not_attempted} question(s) were not attempted.")
    if failures or not_attempted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
