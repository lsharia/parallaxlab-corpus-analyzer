"""Clean the raw RAG corpus and write a validated Parquet dataset."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RAW_PATH = "data/raw/ag_news_train_5000.jsonl"
DEFAULT_OUTPUT_PATH = "data/processed/clean_corpus.parquet"
LANGUAGE = "en"
LANGUAGE_MIN_PROBABILITY = 0.80


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text while ignoring HTML tags and comments."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_html(text: Any) -> str:
    """Remove HTML markup and decode character references from a value."""
    if not isinstance(text, str):
        return ""

    parser = _HTMLTextExtractor()
    try:
        parser.feed(text)
        parser.close()
        return html.unescape("".join(parser.parts))
    except Exception:
        # Malformed markup should not discard an otherwise usable document.
        return html.unescape(re.sub(r"<[^>]*>", " ", text))


def normalize_unicode(text: Any) -> str:
    """Apply a consistent Unicode NFC normalization to string input."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: Any) -> str:
    """Collapse runs of whitespace and trim the document boundaries."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_valid_language(
    text: Any,
    language: str = LANGUAGE,
    min_probability: float = LANGUAGE_MIN_PROBABILITY,
) -> bool:
    """Return whether text is confidently identified as the requested language."""
    if not isinstance(text, str) or not text.strip():
        return False

    try:
        from langdetect import DetectorFactory, detect_langs
    except ImportError as error:
        raise RuntimeError(
            "The 'langdetect' package is required. Install requirements.txt first."
        ) from error

    DetectorFactory.seed = 0
    try:
        language_scores = detect_langs(text)
    except Exception:
        return False

    return any(
        result.lang == language and result.prob >= min_probability
        for result in language_scores
    )


def clean_text(text: Any) -> str:
    """Run the deterministic text-cleaning operations in a fixed order."""
    cleaned = clean_html(text)
    cleaned = normalize_unicode(cleaned)
    return clean_whitespace(cleaned)


def load_raw_dataset(raw_path: Path) -> list[dict[str, Any]]:
    """Load raw JSONL records without changing their source metadata or text."""
    records: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            records.append(record)
    return records


def process_records(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Clean records, filter invalid documents, and return processing statistics."""
    source_records = list(records)
    cleaned_records: list[dict[str, Any]] = []
    empty_documents = 0
    language_filtered_documents = 0
    before_lengths: list[int] = []
    after_lengths: list[int] = []

    for record in source_records:
        raw_text = record.get("text")
        if isinstance(raw_text, str):
            before_lengths.append(len(raw_text))

        cleaned = clean_text(raw_text)
        if not cleaned:
            empty_documents += 1
            continue

        if not is_valid_language(cleaned):
            language_filtered_documents += 1
            continue

        output_record = dict(record)
        output_record["text"] = cleaned
        cleaned_records.append(output_record)
        after_lengths.append(len(cleaned))

    raw_count = len(source_records)
    return cleaned_records, {
        "raw_documents": raw_count,
        "cleaned_documents": len(cleaned_records),
        "removed_documents": raw_count - len(cleaned_records),
        "empty_documents_removed": empty_documents,
        "language_filtered_documents": language_filtered_documents,
        "average_text_length_before": (
            sum(before_lengths) / len(before_lengths) if before_lengths else 0.0
        ),
        "average_text_length_after": (
            sum(after_lengths) / len(after_lengths) if after_lengths else 0.0
        ),
    }


def save_clean_dataset(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write cleaned records to Parquet with a stable column order."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("The 'pandas' package is required.") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records)
    preferred_columns = [
        "document_id",
        "source",
        "split",
        "title",
        "text",
        "label",
        "source_identifier",
    ]
    columns = [column for column in preferred_columns if column in dataframe.columns]
    remaining_columns = [
        column for column in dataframe.columns if column not in columns
    ]
    dataframe = dataframe[columns + remaining_columns]
    dataframe.to_parquet(output_path, index=False)


def print_statistics(statistics: dict[str, float | int]) -> None:
    """Print the required processing statistics in a readable format."""
    print("\nPreprocessing statistics:")
    print(f"Raw documents: {statistics['raw_documents']:,}")
    print(f"Cleaned documents: {statistics['cleaned_documents']:,}")
    print(f"Removed documents: {statistics['removed_documents']:,}")
    print(f"Empty documents removed: {statistics['empty_documents_removed']:,}")
    print(
        "Language-filtered documents: "
        f"{statistics['language_filtered_documents']:,}"
    )
    print(
        "Average text length before cleaning: "
        f"{statistics['average_text_length_before']:.1f} characters"
    )
    print(
        "Average text length after cleaning: "
        f"{statistics['average_text_length_after']:.1f} characters"
    )


def parse_args() -> argparse.Namespace:
    """Parse optional input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Load, clean, validate, and save the raw corpus."""
    project_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    raw_path = (args.raw_path or project_root / DEFAULT_RAW_PATH).resolve()
    output_path = (args.output_path or project_root / DEFAULT_OUTPUT_PATH).resolve()

    try:
        records = load_raw_dataset(raw_path)
        cleaned_records, statistics = process_records(records)
        if not cleaned_records:
            raise ValueError("No documents remained after preprocessing.")
        save_clean_dataset(cleaned_records, output_path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Preprocessing failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print_statistics(statistics)
    print(f"\nClean dataset written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
