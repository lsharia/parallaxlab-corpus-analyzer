"""Download and validate a reproducible raw AG News corpus.

One AG News article is treated as one document. The raw output is JSON Lines so
records can be inspected and processed incrementally in later phases.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


DATASET_NAME = "fancyzhx/ag_news"
DATASET_SPLIT = "train"
DOCUMENT_COUNT = 5_000
OUTPUT_FILENAME = "ag_news_train_5000.jsonl"
SAMPLE_COUNT = 3
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse optional settings while keeping the standard corpus reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to data/raw/ag_news_train_5000.jsonl.",
    )
    return parser.parse_args()


def default_output_path() -> Path:
    """Return the raw-data path relative to the project root."""
    return Path(__file__).resolve().parents[1] / "data" / "raw" / OUTPUT_FILENAME


def load_articles() -> Any:
    """Load the selected public dataset split through the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements.txt first."
        ) from error

    print(f"Loading {DATASET_NAME!r}, split={DATASET_SPLIT!r}...")
    return load_dataset(DATASET_NAME, split=DATASET_SPLIT)


def make_record(article: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert one source row into the stable raw-corpus record format."""
    text = str(article.get("text", ""))
    return {
        "document_id": f"ag_news_{DATASET_SPLIT}_{index:06d}",
        "source": DATASET_NAME,
        "split": DATASET_SPLIT,
        "title": None,
        "text": text,
        "label": article.get("label"),
        "source_identifier": f"{DATASET_NAME}:{DATASET_SPLIT}:{index}",
    }


def write_corpus(dataset: Any, output_path: Path) -> int:
    """Write the first 5,000 source rows as JSONL without cleaning their text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, article in enumerate(dataset.select(range(DOCUMENT_COUNT))):
            record = make_record(article, index)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (index + 1) % 1_000 == 0 or index + 1 == DOCUMENT_COUNT:
                print(f"Saved {index + 1:,}/{DOCUMENT_COUNT:,} documents")
    return DOCUMENT_COUNT


def read_records(output_path: Path) -> list[dict[str, Any]]:
    """Read the JSONL corpus for validation and sample inspection."""
    with output_path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def validate_corpus(records: list[dict[str, Any]]) -> None:
    """Verify count, identifiers, and non-empty raw text before reporting success."""
    if len(records) < DOCUMENT_COUNT:
        raise RuntimeError(
            f"Corpus contains {len(records):,} documents; "
            f"at least {DOCUMENT_COUNT:,} are required."
        )

    missing_text = sum(not isinstance(record.get("text"), str) or not record["text"].strip() for record in records)
    missing_ids = sum(not record.get("document_id") for record in records)
    if missing_text or missing_ids:
        raise RuntimeError(
            f"Corpus validation failed: {missing_text} missing/empty text records, "
            f"{missing_ids} missing document IDs."
        )


def print_statistics(records: list[dict[str, Any]]) -> None:
    """Print corpus statistics and deterministic examples for manual inspection."""
    text_lengths = [len(record["text"]) for record in records]
    print("\nCorpus verification:")
    print(f"Format: JSON Lines (UTF-8), one AG News article per line")
    print(f"Documents: {len(records):,}")
    print(f"Missing/empty text: 0")
    print(f"Text length (characters): min={min(text_lengths):,}, max={max(text_lengths):,}, average={sum(text_lengths) / len(text_lengths):,.1f}")

    sample_records = random.Random(RANDOM_SEED).sample(records, SAMPLE_COUNT)
    print(f"\nRandom examples (seed={RANDOM_SEED}):")
    for record in sample_records:
        preview = " ".join(record["text"].split())[:180]
        print(f"- {record['document_id']}: {preview}")


def main() -> int:
    """Acquire, validate, and summarize the raw corpus."""
    output_path = parse_args().output or default_output_path()
    output_path = output_path.resolve()

    if output_path.exists():
        print(f"Found existing corpus at {output_path}")
        records = read_records(output_path)
    else:
        try:
            dataset = load_articles()
            write_corpus(dataset, output_path)
            records = read_records(output_path)
        except Exception as error:
            print(f"Acquisition failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1

    try:
        validate_corpus(records)
        print_statistics(records)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as error:
        print(f"Corpus validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"\nRaw corpus ready: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
