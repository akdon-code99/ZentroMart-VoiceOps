"""Persistent local NumPy vector index for VoiceOps evidence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from knowledge import EvidenceRecord, load_knowledge_records

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / ".cache" / "rag_index"
INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True)
class VectorIndex:
    records: list[EvidenceRecord]
    vectors: np.ndarray
    manifest: dict[str, Any]


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def save_vector_index(
    index_dir: Path,
    records: list[EvidenceRecord],
    vectors: np.ndarray,
    model_name: str,
) -> VectorIndex:
    """Persist aligned float vectors, evidence records, and a format manifest."""
    index_dir = Path(index_dir)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("Vector rows must match the number of evidence records.")
    if not np.isfinite(matrix).all():
        raise ValueError("Index vectors must contain only finite values.")

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_path = index_dir / "vectors.npy"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=index_dir,
        prefix=".vectors.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_vector = Path(handle.name)
        np.save(handle, matrix, allow_pickle=False)
    os.replace(temporary_vector, vector_path)

    record_data = [
        {
            "evidence_id": record.evidence_id,
            "text": record.text,
            "source_text": record.source_text,
            "metadata": record.metadata,
        }
        for record in records
    ]
    _atomic_write_json(index_dir / "records.json", record_data)
    manifest = {
        "format_version": INDEX_FORMAT_VERSION,
        "model_name": model_name,
        "record_count": len(records),
        "vector_dimensions": int(matrix.shape[1]),
    }
    _atomic_write_json(index_dir / "manifest.json", manifest)
    return VectorIndex(records, matrix, manifest)


def load_vector_index(index_dir: Path) -> VectorIndex:
    """Load and validate a persisted local index."""
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError("Unsupported RAG index format version.")
    raw_records = json.loads((index_dir / "records.json").read_text(encoding="utf-8"))
    records = [
        EvidenceRecord(
            evidence_id=item["evidence_id"],
            text=item["text"],
            source_text=item["source_text"],
            metadata=item["metadata"],
        )
        for item in raw_records
    ]
    vectors = np.load(index_dir / "vectors.npy", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError("Stored vectors do not align with stored evidence records.")
    if (
        manifest.get("record_count") != len(records)
        or manifest.get("vector_dimensions") != vectors.shape[1]
    ):
        raise ValueError("Stored index manifest does not match its files.")
    return VectorIndex(records, vectors, manifest)


def build_vector_index(
    project_root: Path,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    model: Any | None = None,
) -> VectorIndex:
    """Rebuild the local index from source using passage-prefixed E5 embeddings."""
    if model is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name)
    else:
        encoder = model
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        first_module = encoder[0]
        tokenizer = getattr(first_module, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The embedding model does not expose its tokenizer.")

    def count_passage_tokens(text: str) -> int:
        tokens = tokenizer(
            f"passage: {text}",
            add_special_tokens=True,
            truncation=False,
        )
        return len(tokens["input_ids"])

    records = load_knowledge_records(
        project_root,
        max_tokens=500,
        token_counter=count_passage_tokens,
    )
    passages = [f"passage: {record.text}" for record in records]
    vectors = encoder.encode(
        passages,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return save_vector_index(index_dir, records, vectors, model_name)
