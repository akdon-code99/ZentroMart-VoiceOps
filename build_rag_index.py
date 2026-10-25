"""Build or rebuild the local RAG vector index from supplied source files."""

import argparse
from pathlib import Path

from vector_index import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL_NAME,
    build_vector_index,
)

_PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"local index directory (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformers model used for indexing",
    )
    args = parser.parse_args()

    index = build_vector_index(
        project_root=_PROJECT_ROOT,
        index_dir=args.index_dir,
        model_name=args.model,
    )
    print(
        f"Indexed {len(index.records)} evidence records "
        f"({index.vectors.shape[1]} dimensions) at {args.index_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
