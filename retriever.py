"""Multilingual cosine retrieval with source and policy metadata filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from vector_index import VectorIndex

_AUTHORITY_ORDER = {
    "meta": 0,
    "policy": 1,
    "csv": 2,
    "agent_guide": 3,
    "help": 4,
    "faq": 5,
    "marketing": 6,
}
_SKU_PATTERN = re.compile(r"\b[A-Z]{2,5}-\d{4}\b", re.IGNORECASE)
_MAX_AUTHORITY_ADJUSTMENT = 0.03
_EVENT_DATE_PATTERN = re.compile(
    r"\b(?P<event>delivered|ordered|purchased)\s+on\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
_HISTORICAL_SCOPE_PATTERN = re.compile(
    r"\b(?P<category>electronics(?:\s+and\s+accessories)?)\s+"
    r"\*{0,2}(?P<event>delivered)\s+before\s+"
    r"(?P<cutoff>\d{1,2}\s+[A-Za-z]+\s+\d{4})\*{0,2}"
    r".{0,160}?\bkeep the earlier (?P<days>\d+)-day window "
    r"from version (?P<version>\d+)\b",
    re.IGNORECASE | re.DOTALL,
)
_REFUND_PATTERN = re.compile(
    r"\brefund(?:s|ed|ing)?\b|\bmoney\s+back\b|\bget\s+my\s+money\b|"
    r"रिफ़?ंड|वापस\s+(?:पैसे|रिफ़?ंड)",
    re.IGNORECASE,
)
_RETURN_PATTERN = re.compile(
    r"\breturn(?:s|ed|ing)?\b|\breturnable\b|\bsend it back\b|"
    r"\bbring it back\b|रिटर्न|वापस|लौटा|लौटाने",
    re.IGNORECASE,
)
_DAMAGE_PATTERN = re.compile(
    r"\bdamag(?:e|ed|ing)\b|\bcrack(?:ed)?\b|\bbroken\b|\btorn\b|"
    r"\bdefective\b|\bwrong item\b|\bwrong product\b|खराब|टूटा|फटा",
    re.IGNORECASE,
)
_PICKUP_FEE_PATTERN = re.compile(
    r"\b(?:reverse|pickup|collection)\s+(?:pickup\s+)?(?:fee|charge)\b|"
    r"\bfee\b.{0,40}\b(?:pickup|collection)\b|"
    r"\b(?:pay|charged)\b.{0,40}\b(?:pickup|collection)\b|"
    r"\b(?:pickup|collection)\b.{0,40}\b(?:pay|charged|fee)\b|"
    r"\b(?:pickup|collection)\s+(?:is\s+)?free\b|"
    r"\bfree\s+(?:reverse\s+)?(?:pickup|collection)\b|"
    r"पिकअप.{0,12}(?:फ़ीस|फीस|शुल्क)",
    re.IGNORECASE,
)
_WARRANTY_PATTERN = re.compile(
    r"\bwarrant(?:y|ies)\b|\bassured\b|वारंटी|गारंटी",
    re.IGNORECASE,
)
_CONDITION_PATTERN = re.compile(
    r"\bopen(?:ed)?\b|\bunopened\b|\bseal(?:ed)?\b|\bused\b|"
    r"\bcondition\b|सील|खोल|खुला",
    re.IGNORECASE,
)
_COD_PATTERN = re.compile(r"\bcash[\s-]*on[\s-]*delivery\b|\bcod\b", re.IGNORECASE)
_UNRECOGNIZED_ORDER_PATTERN = re.compile(
    r"\b(?:never|did not|didn't|have not|haven't)\s+"
    r"(?:place(?:d)?|order(?:ed)?|make)\b|"
    r"\b(?:order|payment|account activity)\s+i\s+did(?: not|n't)\b|"
    r"\bunauthori[sz]ed\b|\bnot my order\b",
    re.IGNORECASE,
)
_DATED_QUESTION_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
    re.IGNORECASE,
)
_SPECIFIC_PAYMENT_METHOD_PATTERN = re.compile(
    r"\bupi\b|\bnet[\s-]*banking\b|\bcredit\s+card\b|\bdebit\s+card\b|"
    r"\bzentromart\s+wallet\b|\bcash[\s-]*on[\s-]*delivery\b|\bcod\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalHit:
    evidence_id: str
    text: str
    source_text: str
    metadata: dict[str, Any]
    score: float


class EvidenceRetriever:
    """Retrieve current evidence while retaining its authority/provenance."""

    def __init__(self, index: VectorIndex, encoder: Any) -> None:
        self._index = index
        self._encoder = encoder

    def _eligible_indices(self) -> list[int]:
        sectioned_sources = {
            record.metadata.get("source_path")
            for record in self._index.records
            if (
                record.metadata.get("source_type") != "csv"
                and record.metadata.get("status") == "current"
                and record.metadata.get("source_path")
                and record.metadata.get("section_id")
            )
        }
        eligible = []
        for index, record in enumerate(self._index.records):
            metadata = record.metadata
            if metadata.get("source_type") == "csv":
                eligible.append(index)
            elif metadata.get("status") == "current":
                if (
                    not metadata.get("section_id")
                    and metadata.get("source_path") in sectioned_sources
                ):
                    continue
                eligible.append(index)
        return eligible

    @staticmethod
    def _authority_rank(metadata: dict[str, Any]) -> int:
        if metadata.get("source_type") == "csv":
            return _AUTHORITY_ORDER["csv"]
        return _AUTHORITY_ORDER.get(str(metadata.get("doc_type")), 99)

    def _matching_product_indices(
        self,
        query: str,
        selected: list[tuple[int, float]],
    ) -> list[int]:
        query_folded = query.casefold()
        sku_values = {value.upper() for value in _SKU_PATTERN.findall(query)}
        matched = []
        for index, record in enumerate(self._index.records):
            metadata = record.metadata
            if (
                metadata.get("source_type") != "csv"
                or metadata.get("csv_table") != "products"
            ):
                continue
            record_key = str(metadata.get("record_key", "")).upper()
            row = metadata.get("row")
            title = str(row.get("title", "")).casefold() if isinstance(row, dict) else ""
            title_without_variant = title.split(",", maxsplit=1)[0].strip()
            if (
                record_key in sku_values
                or (title and title in query_folded)
                or (title_without_variant and title_without_variant in query_folded)
            ):
                matched.append(index)
        if matched:
            return matched
        return [
            index
            for index, _ in selected
            if self._index.records[index].metadata.get("source_type") == "csv"
            and self._index.records[index].metadata.get("csv_table") == "products"
        ]

    def _required_section_ids(
        self,
        query: str,
        product_indices: list[int],
    ) -> list[str]:
        required: list[str] = []
        folded = query.casefold()
        is_refund = bool(_REFUND_PATTERN.search(query))
        is_return = bool(_RETURN_PATTERN.search(query))
        is_damage = bool(_DAMAGE_PATTERN.search(query))
        products = [
            self._index.records[index].metadata.get("row", {})
            for index in product_indices
        ]

        if is_refund:
            required.append("REF-2")
            if _COD_PATTERN.search(query):
                required.append("REF-3")
            if _DATED_QUESTION_PATTERN.search(query) or re.search(
                r"\bbusiness days?\b|working days?|کاروباری دن",
                query,
                re.IGNORECASE,
            ):
                required.extend(("KB-00.2", "KB-00.3"))

        if _WARRANTY_PATTERN.search(query):
            required.extend(("WAR-1", "WAR-2"))

        if is_return:
            required.append("RET-2")
            if _CONDITION_PATTERN.search(query):
                required.append("RET-4")

        if is_damage:
            required.extend(("DWM-1", "DWM-2"))
            if _PICKUP_FEE_PATTERN.search(query):
                required.append("RET-6")
            high_value = any(
                isinstance(row, dict)
                and str(row.get("price_inr", "")).replace(",", "").isdigit()
                and int(str(row.get("price_inr", "")).replace(",", "")) > 10000
                for row in products
            )
            if high_value:
                required.extend(("DWM-3", "ESC-1", "ESC-2"))

        if _UNRECOGNIZED_ORDER_PATTERN.search(folded):
            required.extend(("ESC-1", "ESC-2"))

        return list(dict.fromkeys(required))

    def _required_policy_indices(
        self,
        section_ids: list[str],
        query_vector: np.ndarray,
    ) -> list[tuple[int, float]]:
        required = []
        for section_id in section_ids:
            candidates = [
                index
                for index, record in enumerate(self._index.records)
                if record.metadata.get("section_id") == section_id
                and record.metadata.get("status") == "current"
                and record.metadata.get("source_type") != "csv"
            ]
            if not candidates:
                continue
            best_index = max(
                candidates,
                key=lambda index: float(self._index.vectors[index] @ query_vector),
            )
            required.append((
                best_index,
                float(self._index.vectors[best_index] @ query_vector),
            ))
        return required

    @staticmethod
    def _parse_policy_date(value: str, default_year: int | None = None) -> date | None:
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass

        match = re.fullmatch(
            r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)(?:\s+(?P<year>\d{4}))?",
            value.strip(),
        )
        if not match:
            return None
        year_text = match.group("year")
        year = int(year_text) if year_text else default_year
        if year is None:
            return None
        date_text = f"{match.group('day')} {match.group('month')} {year}"
        for date_format in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(date_text, date_format).date()
            except ValueError:
                continue
        return None

    def _frozen_reference_date(self) -> date | None:
        for record in self._index.records:
            if record.metadata.get("section_id") != "KB-00.2":
                continue
            match = re.search(
                r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
                record.source_text,
            )
            if match:
                return self._parse_policy_date(match.group(1))
        return None

    def _historical_policy_pair(
        self, query: str
    ) -> tuple[int, int] | None:
        event_match = _EVENT_DATE_PATTERN.search(query)
        reference_date = self._frozen_reference_date()
        if not event_match or reference_date is None:
            return None

        event_date = self._parse_policy_date(
            event_match.group("date"),
            default_year=reference_date.year,
        )
        if event_date is None:
            return None
        if event_date > reference_date and not re.search(
            r"\b\d{4}\b", event_match.group("date")
        ):
            try:
                event_date = event_date.replace(year=reference_date.year - 1)
            except ValueError:
                return None

        query_folded = query.casefold()
        matching_products = []
        for record in self._index.records:
            if (
                record.metadata.get("source_type") != "csv"
                or record.metadata.get("csv_table") != "products"
                or not isinstance(record.metadata.get("row"), dict)
            ):
                continue
            title = str(record.metadata["row"].get("title", "")).casefold()
            title_variants = {title, title.split(",", maxsplit=1)[0].strip()}
            if any(variant and variant in query_folded for variant in title_variants):
                matching_products.append(record)
        if len(matching_products) != 1:
            return None
        product_category = str(
            matching_products[0].metadata["row"].get("return_category", "")
        ).casefold()

        for successor_index, successor in enumerate(self._index.records):
            successor_metadata = successor.metadata
            if (
                successor_metadata.get("source_type") == "csv"
                or successor_metadata.get("doc_type") != "policy"
                or successor_metadata.get("status") != "current"
                or not successor_metadata.get("supersedes")
            ):
                continue
            scope = _HISTORICAL_SCOPE_PATTERN.search(successor.source_text)
            if not scope or event_match.group("event").casefold() != scope.group(
                "event"
            ).casefold():
                continue
            category = scope.group("category").casefold()
            if product_category not in category and category not in product_category:
                continue

            cutoff = self._parse_policy_date(scope.group("cutoff"))
            effective_from = self._parse_policy_date(
                str(successor_metadata.get("effective_from", ""))
            )
            if cutoff is None or cutoff != effective_from or event_date >= cutoff:
                continue
            version = scope.group("version")
            days_text = f"{scope.group('days')} days"
            successor_id = str(successor_metadata.get("doc_id", ""))

            for historical_index, historical in enumerate(self._index.records):
                metadata = historical.metadata
                if (
                    metadata.get("status") != "superseded"
                    or metadata.get("superseded_by") != successor_id
                    or str(metadata.get("version", "")).split(".", maxsplit=1)[0]
                    != version
                ):
                    continue
                historical_effective = self._parse_policy_date(
                    str(metadata.get("effective_from", ""))
                )
                if historical_effective is None or event_date < historical_effective:
                    continue
                for line in historical.source_text.splitlines():
                    cells = [
                        cell.strip().casefold()
                        for cell in line.strip().strip("|").split("|")
                    ]
                    if (
                        len(cells) >= 2
                        and category in cells[0]
                        and days_text in cells[1]
                    ):
                        return successor_index, historical_index
        return None

    def search(
        self,
        query: str,
        top_k: int = 8,
        min_score: float | None = None,
        candidate_pool: int | None = None,
        authority_adjustment: float = 0.03,
        relevance_window: float = 0.002,
        max_product_rows: int = 2,
    ) -> list[RetrievalHit]:
        """Search with E5, optionally using bounded relevance-aware reranking."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if candidate_pool is not None and candidate_pool < top_k:
            raise ValueError("candidate_pool must be at least top_k.")
        if not 0 <= authority_adjustment <= _MAX_AUTHORITY_ADJUSTMENT:
            raise ValueError(
                f"authority_adjustment must be between 0 and "
                f"{_MAX_AUTHORITY_ADJUSTMENT}."
            )
        if relevance_window < 0:
            raise ValueError("relevance_window must not be negative.")
        if max_product_rows < 1:
            raise ValueError("max_product_rows must be at least 1.")

        query_vector = self._encoder.encode(
            [f"query: {query.strip()}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query_vector.shape[0] != self._index.vectors.shape[1]:
            raise ValueError("Query embedding dimensions do not match the index.")

        eligible = self._eligible_indices()
        if not eligible:
            return []
        scores = self._index.vectors[eligible] @ query_vector
        ranked = sorted(
            (
                (int(record_index), float(score))
                for record_index, score in zip(eligible, scores, strict=True)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if min_score is not None:
            ranked = [item for item in ranked if item[1] >= min_score]

        def adjusted_score(item: tuple[int, float]) -> float:
            metadata = self._index.records[item[0]].metadata
            doc_type = metadata.get("doc_type")
            if doc_type == "policy":
                adjustment = authority_adjustment
            elif doc_type == "agent_guide":
                adjustment = authority_adjustment / 2
            else:
                adjustment = 0.0
            return item[1] + adjustment

        if candidate_pool is None:
            selected = ranked[:top_k]
        else:
            candidates = ranked[:candidate_pool]
            if candidates:
                cutoff = ranked[min(top_k - 1, len(ranked) - 1)][1]
                candidates = [
                    item
                    for item in candidates
                    if item[1] >= cutoff - relevance_window
                ]

            candidates.sort(
                key=lambda item: (adjusted_score(item), item[1]),
                reverse=True,
            )
            selected = []
            product_rows = 0
            for item in candidates:
                metadata = self._index.records[item[0]].metadata
                is_product_row = (
                    metadata.get("source_type") == "csv"
                    and metadata.get("csv_table") == "products"
                )
                if is_product_row and product_rows >= max_product_rows:
                    continue
                selected.append(item)
                product_rows += int(is_product_row)
                if len(selected) == top_k:
                    break

        selected_ids = {self._index.records[index].evidence_id for index, _ in selected}
        exact_skus = {sku.upper() for sku in _SKU_PATTERN.findall(query)}
        exact_matches = []
        for index in eligible:
            record = self._index.records[index]
            if (
                record.metadata.get("source_type") == "csv"
                and record.metadata.get("csv_table") == "products"
                and str(record.metadata.get("record_key", "")).upper() in exact_skus
                and record.evidence_id not in selected_ids
            ):
                score = float(self._index.vectors[index] @ query_vector)
                exact_matches.append((index, score))
                selected_ids.add(record.evidence_id)

        for exact_match in exact_matches:
            if len(selected) == top_k:
                removable = [
                    (position, item)
                    for position, item in enumerate(selected)
                    if item[0] not in {index for index, _ in exact_matches}
                ]
                if not removable:
                    break
                remove_position, _ = min(removable, key=lambda pair: pair[1][1])
                selected.pop(remove_position)
            selected.append(exact_match)

        if (
            _REFUND_PATTERN.search(query)
            and _SPECIFIC_PAYMENT_METHOD_PATTERN.search(query)
        ):
            selected = [
                item
                for item in selected
                if self._index.records[item[0]].metadata.get("doc_type") != "faq"
            ]

        matching_products = self._matching_product_indices(query, selected)[
            :max_product_rows
        ]
        required_sections = self._required_section_ids(query, matching_products)
        required = [
            (index, float(self._index.vectors[index] @ query_vector))
            for index in matching_products
        ]
        required_policy = self._required_policy_indices(
            required_sections,
            query_vector,
        )
        required.extend(required_policy)
        mandatory_ids = {index for index, _ in required}
        product_rows = len(matching_products)
        for item in selected:
            if item[0] in mandatory_ids:
                continue
            metadata = self._index.records[item[0]].metadata
            is_product_row = (
                metadata.get("source_type") == "csv"
                and metadata.get("csv_table") == "products"
            )
            if is_product_row and product_rows >= max_product_rows:
                continue
            required.append(item)
            product_rows += int(is_product_row)
            mandatory_ids.add(item[0])
        selected = required[:top_k]

        if candidate_pool is None:
            selected.sort(
                key=lambda item: (
                    self._authority_rank(self._index.records[item[0]].metadata),
                    -item[1],
                )
            )
        else:
            selected.sort(
                key=lambda item: (adjusted_score(item), item[1]),
                reverse=True,
            )
        required_policy_indices = {index for index, _ in required_policy}
        hits = [
            RetrievalHit(
                evidence_id=self._index.records[index].evidence_id,
                text=self._index.records[index].text,
                source_text=self._index.records[index].source_text,
                metadata=(
                    {
                        **self._index.records[index].metadata,
                        "retrieval_role": "intent_routed_policy",
                    }
                    if index in required_policy_indices
                    else self._index.records[index].metadata
                ),
                score=score,
            )
            for index, score in selected
        ]
        historical_pair = self._historical_policy_pair(query)
        if historical_pair is None or top_k < 2:
            return hits

        successor_index, historical_index = historical_pair
        successor = self._index.records[successor_index]
        historical = self._index.records[historical_index]
        protected_ids = {successor.evidence_id, historical.evidence_id}
        remaining = [
            hit for hit in hits if hit.evidence_id not in protected_ids
        ]
        successor_hit = RetrievalHit(
            evidence_id=successor.evidence_id,
            text=successor.text,
            source_text=successor.source_text,
            metadata={**successor.metadata, "retrieval_role": "governing_current_policy"},
            score=float(self._index.vectors[successor_index] @ query_vector),
        )
        historical_hit = RetrievalHit(
            evidence_id=historical.evidence_id,
            text=(
                "[HISTORICAL POLICY REFERENCE — NOT GOVERNING CURRENT POLICY]\n"
                f"{historical.text}"
            ),
            source_text=historical.source_text,
            metadata={**historical.metadata, "retrieval_role": "historical_reference"},
            score=float(self._index.vectors[historical_index] @ query_vector),
        )
        return [successor_hit, historical_hit, *remaining[: top_k - 2]]

    def authority_guidance(self) -> str:
        """Return the supplied KB precedence section for answer-prompt guidance."""
        for record in self._index.records:
            if (
                record.metadata.get("doc_id") == "KB-00"
                and record.metadata.get("section_id") == "KB-00.1"
                and record.metadata.get("status") == "current"
            ):
                return record.source_text
        raise ValueError("Current KB precedence guidance KB-00.1 is missing.")
