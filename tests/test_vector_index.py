import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vector_index import (
    build_vector_index,
    load_vector_index,
    save_vector_index,
)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


class FakeEncoder:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.encoded_texts = []

    def encode(self, texts, **kwargs):
        self.encoded_texts.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class VectorIndexTests(unittest.TestCase):
    def test_save_and_load_preserve_vectors_text_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            from knowledge import EvidenceRecord

            record = EvidenceRecord(
                evidence_id="kb/policy.md::RET-1::0::abc",
                text="## [RET-1] A policy section",
                source_text="## [RET-1] A policy section\nOriginal body.",
                metadata={
                    "source_path": "kb/policy.md",
                    "doc_id": "POL-RET",
                    "section_id": "RET-1",
                    "version": "4.0",
                    "status": "current",
                },
            )
            saved = save_vector_index(
                Path(directory),
                [record],
                np.asarray([[0.6, 0.8]], dtype=np.float32),
                "intfloat/multilingual-e5-small",
            )
            loaded = load_vector_index(Path(directory))

        self.assertEqual(saved.manifest["record_count"], 1)
        self.assertEqual(loaded.records[0].source_text, record.source_text)
        self.assertEqual(loaded.records[0].metadata["version"], "4.0")
        np.testing.assert_allclose(loaded.vectors, [[0.6, 0.8]])

    def test_build_recreates_deleted_index_from_sources_and_uses_passage_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kb = root / "kb"
            data = root / "data"
            index_dir = root / ".cache" / "rag_index"
            kb.mkdir()
            data.mkdir()
            (kb / "policy.md").write_text(
                "---\ndoc_id: POL-X\nstatus: current\ndoc_type: policy\n---\n"
                "## [POL-X-1] Rule\nThe supplied source rule.\n",
                encoding="utf-8",
            )
            for filename, fields, row in (
                ("products.csv", ["sku"], ["ELE-0012"]),
                ("pincodes.csv", ["pincode"], ["560001"]),
                ("holidays.csv", ["date"], ["2026-10-19"]),
            ):
                with (data / filename).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(fields)
                    writer.writerow(row)

            first_encoder = FakeEncoder()
            first = build_vector_index(
                root,
                index_dir,
                model_name="fake",
                model=first_encoder,
            )
            self.assertEqual(first.manifest["record_count"], 4)
            self.assertTrue(all(
                text.startswith("passage: ") for text in first_encoder.encoded_texts
            ))

            (index_dir / "vectors.npy").unlink()
            rebuilt_encoder = FakeEncoder()
            rebuilt = build_vector_index(
                root,
                index_dir,
                model_name="fake",
                model=rebuilt_encoder,
            )
            loaded = load_vector_index(index_dir)
            vectors_exist = (index_dir / "vectors.npy").exists()

        self.assertEqual(rebuilt.manifest["record_count"], 4)
        self.assertEqual(loaded.manifest["model_name"], "fake")
        self.assertEqual(len(loaded.records), 4)
        self.assertTrue(vectors_exist)


if __name__ == "__main__":
    unittest.main()
