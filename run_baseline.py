"""Run the LLM-only baseline over practice/questions.csv."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from llm_client import OllamaCloudError, baseline_answer, validate_configuration

_PROJECT_ROOT = Path(__file__).resolve().parent
_QUESTIONS_PATH = _PROJECT_ROOT / "practice" / "questions.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "baseline_results.csv"


def _positive_int(value: str) -> int:
    limit = int(value)
    if limit < 1:
        raise argparse.ArgumentTypeError("limit must be at least 1")
    return limit


def run_questions(
    questions: list[dict[str, str]], limit: int | None = None
) -> list[dict[str, str]]:
    """Generate a result row per input while retaining all evaluation columns."""
    selected = questions if limit is None else questions[:limit]
    results = []

    for index, row in enumerate(selected, start=1):
        result = dict(row)
        result["generated_answer"] = ""
        result["error"] = ""
        language = row.get("reply_language") or row.get("language")
        try:
            result["generated_answer"] = baseline_answer(
                row["question"],
                reply_language=language,
            )
        except (OllamaCloudError, RuntimeError, ValueError) as error:
            result["error"] = str(error)
            print(
                f"Question {row.get('id', index)} failed: {error}",
                file=sys.stderr,
            )
        results.append(result)
        print(f"Processed {index}/{len(selected)}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="process only the first N questions (each sends one API request)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"result CSV path (default: {_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    try:
        validate_configuration()
        question_frame = pd.read_csv(
            _QUESTIONS_PATH,
            dtype=str,
            keep_default_na=False,
        )
        if "question" not in question_frame.columns:
            raise ValueError("practice/questions.csv has no 'question' column.")
        if "expected_decision" not in question_frame.columns:
            raise ValueError(
                "practice/questions.csv has no 'expected_decision' column."
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Baseline setup failed: {error}", file=sys.stderr)
        return 2

    questions = question_frame.to_dict(orient="records")
    results = run_questions(questions, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=[
        *question_frame.columns,
        "generated_answer",
        "error",
    ]).to_csv(args.output, index=False)

    failures = sum(bool(row["error"]) for row in results)
    print(f"Saved {len(results)} result(s) to {args.output}")
    if failures:
        print(f"{failures} request(s) failed; see the error column.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
