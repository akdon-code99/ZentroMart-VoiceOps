import csv
import tempfile
import unittest
from pathlib import Path

from knowledge import (
    load_markdown_records,
    load_structured_csv_records,
    parse_front_matter,
)


class FrontMatterTests(unittest.TestCase):
    def test_optional_supersession_fields_are_not_required(self):
        source = "---\ndoc_id: DOC-1\nstatus: current\n---\n## [SEC-1] Title\nBody\n"

        metadata, body = parse_front_matter(source)

        self.assertEqual(metadata, {"doc_id": "DOC-1", "status": "current"})
        self.assertIn("## [SEC-1] Title", body)
        self.assertNotIn("supersedes", metadata)

    def test_document_without_front_matter_is_preserved(self):
        source = "# Heading\n\nText\n"

        metadata, body = parse_front_matter(source)

        self.assertEqual(metadata, {})
        self.assertEqual(body, source)

    def test_unclosed_front_matter_is_reported(self):
        with self.assertRaisesRegex(ValueError, "not closed"):
            parse_front_matter("---\ndoc_id: DOC-1\n")


class MarkdownLoadingTests(unittest.TestCase):
    def test_sections_preamble_metadata_and_source_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kb = root / "kb"
            kb.mkdir()
            source_text = (
                "---\n"
                "doc_id: POL-X\n"
                "doc_type: policy\n"
                "status: superseded\n"
                "superseded_by: POL-Y\n"
                "---\n"
                "Preamble that must remain searchable.\n\n"
                "## [SEC-1] Current rule exception\n"
                "Rule condition.\n\n"
                "Exception applies to older cases.\n"
                "## [SEC-2] Another section\n"
                "Second body.\n"
            )
            (kb / "sample.md").write_text(source_text, encoding="utf-8")

            records = load_markdown_records(kb, project_root=root)

        self.assertEqual(len(records), 3)
        preamble, first, second = records
        self.assertIsNone(preamble.metadata["section_id"])
        self.assertIn("Preamble that must remain searchable", preamble.text)
        self.assertEqual(first.metadata["section_id"], "SEC-1")
        self.assertEqual(first.metadata["doc_type"], "policy")
        self.assertEqual(first.metadata["status"], "superseded")
        self.assertEqual(first.metadata["superseded_by"], "POL-Y")
        self.assertIn("Rule condition", first.source_text)
        self.assertIn("Exception applies", first.text)
        self.assertEqual(second.metadata["section_title"], "Another section")
        self.assertTrue(first.metadata["source_path"].endswith("sample.md"))

    def test_long_section_is_split_at_paragraph_boundary_with_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kb = root / "kb"
            kb.mkdir()
            (kb / "long.md").write_text(
                "---\ndoc_id: DOC-LONG\nstatus: current\n---\n"
                "## [LONG-1] Main rule\n"
                "condition exception\n\n"
                "second paragraph\n\n"
                "third paragraph\n",
                encoding="utf-8",
            )

            records = load_markdown_records(
                kb,
                project_root=root,
                max_tokens=6,
                token_counter=lambda text: len(text.split()),
            )

        self.assertGreater(len(records), 1)
        self.assertTrue(all(record.metadata["section_id"] == "LONG-1" for record in records))
        self.assertTrue(all(record.metadata["doc_id"] == "DOC-LONG" for record in records))
        self.assertIn("condition exception", " ".join(record.text for record in records))


class StructuredCsvTests(unittest.TestCase):
    def test_actual_style_product_row_is_source_tagged_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            path = data / "products.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sku", "title", "brand", "return_category", "price_inr",
                        "fulfilled_by", "assured", "warranty_months", "description",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "sku": "ELE-0012",
                    "title": "Kestrel K7 Smartphone, Beige",
                    "brand": "Kestrel",
                    "return_category": "electronics",
                    "price_inr": "12999",
                    "fulfilled_by": "ZentroMart",
                    "assured": "True",
                    "warranty_months": "12",
                    "description": "Synthetic catalog row",
                })

            records = load_structured_csv_records(
                data, project_root=root, filenames=("products.csv",)
            )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.evidence_id, "products.csv:ELE-0012")
        self.assertEqual(record.metadata["source_type"], "csv")
        self.assertEqual(record.metadata["row"]["assured"], "True")
        self.assertIn("warranty_months: 12", record.text)
        self.assertIn("Synthetic catalog row", record.source_text)


if __name__ == "__main__":
    unittest.main()
