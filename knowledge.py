"""Load section-aware Markdown and selected structured CSV evidence."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_SECTION_HEADING = re.compile(r"^(#{2,6})\s+\[([^\]]+)\]\s*(.*?)\s*$")
_FRONTMATTER_FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CSV_PRIMARY_KEYS = {
    "products.csv": "sku",
    "pincodes.csv": "pincode",
    "holidays.csv": "date",
}

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class EvidenceRecord:
    """One retrievable chunk with its source text and provenance metadata."""

    evidence_id: str
    text: str
    source_text: str
    metadata: dict[str, Any]


def parse_front_matter(markdown: str) -> tuple[dict[str, str], str]:
    """Parse the supplied flat scalar front matter without requiring all keys."""
    lines = markdown.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, markdown

    closing_index = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError("Markdown front matter starts with '---' but is not closed.")

    metadata: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:closing_index], start=2):
        value = line.rstrip("\r\n")
        if not value.strip() or value.lstrip().startswith("#"):
            continue
        match = _FRONTMATTER_FIELD.match(value)
        if not match:
            raise ValueError(f"Unsupported front-matter field on line {line_number}.")
        key, field_value = match.groups()
        metadata[key] = field_value.strip().strip("\"'")

    body = "".join(lines[closing_index + 1 :])
    return metadata, body


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n[ \t]*\n+", text) if part.strip()]


def _default_token_count(text: str) -> int:
    """Conservative fallback; index building should pass the model tokenizer."""
    return len(re.findall(r"\S+", text))


def _split_oversized_block(
    block: str,
    prefix: str,
    max_tokens: int,
    token_count: TokenCounter,
) -> list[str]:
    if token_count(f"{prefix}\n{block}") <= max_tokens:
        return [block]

    lines = block.splitlines()
    if len(lines) > 2 and all(line.lstrip().startswith("|") for line in lines):
        header_lines = lines[:2] if _TABLE_SEPARATOR.match(lines[1]) else lines[:1]
        data_lines = lines[len(header_lines) :]
        chunks: list[str] = []
        current: list[str] = []
        for line in data_lines:
            candidate = "\n".join([*header_lines, *current, line])
            if current and token_count(f"{prefix}\n{candidate}") > max_tokens:
                chunks.append("\n".join([*header_lines, *current]))
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("\n".join([*header_lines, *current]))
        return chunks

    sentences = re.split(r"(?<=[.!?।])\s+", block)
    units: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and token_count(f"{prefix}\n{candidate}") > max_tokens:
            units.append(current)
            current = sentence
        else:
            current = candidate
        if token_count(f"{prefix}\n{current}") > max_tokens:
            words = current.split()
            word_chunk = ""
            for word in words:
                candidate = f"{word_chunk} {word}".strip()
                if word_chunk and token_count(f"{prefix}\n{candidate}") > max_tokens:
                    units.append(word_chunk)
                    word_chunk = word
                else:
                    word_chunk = candidate
            current = word_chunk
    if current:
        units.append(current)
    return units


def _chunk_section(
    source_text: str,
    heading: str,
    max_tokens: int,
    token_count: TokenCounter,
) -> list[str]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    prefix = heading.strip()
    if token_count(source_text) <= max_tokens:
        return [source_text]

    body = source_text[len(heading) :].lstrip("\r\n") if heading else source_text
    blocks: list[str] = []
    for paragraph in _paragraphs(body):
        blocks.extend(
            _split_oversized_block(paragraph, prefix, max_tokens, token_count)
        )

    chunks: list[str] = []
    current_blocks: list[str] = []
    for block in blocks:
        candidate = "\n\n".join([*current_blocks, block])
        candidate_text = f"{prefix}\n{candidate}".strip() if prefix else candidate
        if current_blocks and token_count(candidate_text) > max_tokens:
            chunk_body = "\n\n".join(current_blocks)
            chunks.append(f"{prefix}\n{chunk_body}" if prefix else chunk_body)
            previous = current_blocks[-1]
            current_blocks = (
                [previous, block]
                if token_count(f"{prefix}\n{previous}\n\n{block}") <= max_tokens
                else [block]
            )
        else:
            current_blocks.append(block)

    if current_blocks:
        chunk_body = "\n\n".join(current_blocks)
        chunks.append(f"{prefix}\n{chunk_body}" if prefix else chunk_body)
    return chunks or [source_text]


def _relative_source(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _record_id(source_path: str, section_id: str | None, index: int, text: str) -> str:
    identity = section_id or "preamble"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{source_path}::{identity}::{index}::{digest}"


def load_markdown_records(
    kb_dir: Path,
    project_root: Path | None = None,
    max_tokens: int = 450,
    token_counter: TokenCounter | None = None,
) -> list[EvidenceRecord]:
    """Read Markdown docs, preserving preambles, section IDs, source and metadata."""
    kb_dir = Path(kb_dir)
    root = Path(project_root) if project_root else kb_dir.parent
    count_tokens = token_counter or _default_token_count
    records: list[EvidenceRecord] = []

    for path in sorted(kb_dir.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        front_matter, body = parse_front_matter(markdown)
        source_path = _relative_source(path, root)
        lines = body.splitlines(keepends=True)
        headings = [
            (index, match)
            for index, line in enumerate(lines)
            if (match := _SECTION_HEADING.match(line.rstrip("\r\n")))
        ]
        segments: list[tuple[str | None, str | None, str]] = []
        first_heading_line = headings[0][0] if headings else len(lines)
        preamble = "".join(lines[:first_heading_line])
        if preamble.strip():
            segments.append((None, None, preamble))
        for heading_index, (line_index, match) in enumerate(headings):
            end_index = (
                headings[heading_index + 1][0]
                if heading_index + 1 < len(headings)
                else len(lines)
            )
            section_source = "".join(lines[line_index:end_index])
            segments.append((match.group(2), match.group(3).strip(), section_source))
        if not segments and body.strip():
            segments.append((None, None, body))

        for section_id, section_title, source_text in segments:
            heading = source_text.splitlines()[0] if section_id else ""
            chunks = _chunk_section(source_text, heading, max_tokens, count_tokens)
            for chunk_index, chunk_text in enumerate(chunks):
                metadata: dict[str, Any] = {
                    **front_matter,
                    "source_path": source_path,
                    "section_id": section_id,
                    "section_title": section_title,
                    "chunk_index": chunk_index,
                }
                records.append(
                    EvidenceRecord(
                        evidence_id=_record_id(
                            source_path, section_id, chunk_index, chunk_text
                        ),
                        text=chunk_text,
                        source_text=source_text,
                        metadata=metadata,
                    )
                )
    return records


def load_structured_csv_records(
    data_dir: Path,
    project_root: Path | None = None,
    filenames: tuple[str, ...] = tuple(_CSV_PRIMARY_KEYS),
) -> list[EvidenceRecord]:
    """Load product, pincode, and holiday rows as traceable structured evidence."""
    data_dir = Path(data_dir)
    root = Path(project_root) if project_root else data_dir.parent
    records: list[EvidenceRecord] = []

    for filename in filenames:
        primary_key = _CSV_PRIMARY_KEYS.get(filename)
        if primary_key is None:
            raise ValueError(f"Unsupported structured evidence CSV: {filename}")
        path = data_dir / filename
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or primary_key not in reader.fieldnames:
                raise ValueError(
                    f"{filename} must contain its observed key column {primary_key!r}."
                )
            for row in reader:
                key_value = row.get(primary_key, "")
                if not key_value:
                    raise ValueError(f"{filename} contains a blank {primary_key}.")
                source_path = _relative_source(path, root)
                row_text = "\n".join(f"{key}: {value}" for key, value in row.items())
                text = f"Source: {filename} ({primary_key}={key_value})\n{row_text}"
                source_kind = Path(filename).stem
                metadata: dict[str, Any] = {
                    "source_path": source_path,
                    "source_type": "csv",
                    "csv_table": source_kind,
                    "primary_key": primary_key,
                    "record_key": key_value,
                    "section_id": None,
                    "status": None,
                    "row": row,
                }
                records.append(
                    EvidenceRecord(
                        evidence_id=f"{filename}:{key_value}",
                        text=text,
                        source_text=row_text,
                        metadata=metadata,
                    )
                )
    return records


def load_knowledge_records(
    project_root: Path,
    max_tokens: int = 450,
    token_counter: TokenCounter | None = None,
) -> list[EvidenceRecord]:
    """Load all KB sections and the selected structured CSV evidence."""
    project_root = Path(project_root)
    return [
        *load_markdown_records(
            project_root / "kb",
            project_root=project_root,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ),
        *load_structured_csv_records(
            project_root / "data",
            project_root=project_root,
        ),
    ]
