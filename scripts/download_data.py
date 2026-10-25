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
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol


DATASET_NAME: str = "fancyzhx/ag_news"
DATASET_SPLIT: str = "train"
DEFAULT_DOCUMENT_COUNT: int = 5_200
SAMPLE_COUNT: int = 3
RANDOM_SEED: int = 42


class ArticleDataset(Protocol):
    """Minimal dataset interface used to select and iterate source rows."""

    def __len__(self) -> int: ...

    def select(self, indices: Iterable[int]) -> Iterable[Mapping[str, object]]:
        """Return source rows at the requested positions."""
        ...


def parse_args() -> argparse.Namespace:
    """Parse optional settings while keeping the standard corpus reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_DOCUMENT_COUNT,
        help=f"Number of source articles to download (default: {DEFAULT_DOCUMENT_COUNT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to data/raw/ag_news_train_<count>.jsonl.",
    )
    return parser.parse_args()


def default_output_path(document_count: int = DEFAULT_DOCUMENT_COUNT) -> Path:
    """Return the raw-data path relative to the project root."""
    filename = f"ag_news_train_{document_count}.jsonl"
    return Path(__file__).resolve().parents[1] / "data" / "raw" / filename


def load_articles() -> ArticleDataset:
    """Load the selected public dataset split through the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements.txt first."
        ) from error

    print(f"Loading {DATASET_NAME!r}, split={DATASET_SPLIT!r}...")
    return load_dataset(DATASET_NAME, split=DATASET_SPLIT)


def make_record(
    article: Mapping[str, object],
    index: int,
) -> dict[str, object]:
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


def write_corpus(
    dataset: ArticleDataset,
    output_path: Path,
    document_count: int = DEFAULT_DOCUMENT_COUNT,
) -> int:
    """Write the requested initial source rows as raw JSONL records."""
    if document_count <= 0:
        raise ValueError("document_count must be greater than zero")
    if len(dataset) < document_count:
        raise ValueError(
            f"Source split contains {len(dataset):,} rows; "
            f"{document_count:,} were requested."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, article in enumerate(dataset.select(range(document_count))):
            record = make_record(article, index)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (index + 1) % 1_000 == 0 or index + 1 == document_count:
                print(f"Saved {index + 1:,}/{document_count:,} documents")
    return document_count


def read_records(output_path: Path) -> list[dict[str, object]]:
    """Read the JSONL corpus for validation and sample inspection."""
    records: list[dict[str, object]] = []
    with output_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            records.append(record)
    return records


def validate_corpus(
    records: Sequence[Mapping[str, object]],
    minimum_count: int = DEFAULT_DOCUMENT_COUNT,
) -> None:
    """Verify count, identifiers, and non-empty raw text before reporting success."""
    if minimum_count <= 0:
        raise ValueError("minimum_count must be greater than zero")
    if len(records) < minimum_count:
        raise RuntimeError(
            f"Corpus contains {len(records):,} documents; "
            f"at least {minimum_count:,} are required."
        )

    missing_text = sum(not isinstance(record.get("text"), str) or not record["text"].strip() for record in records)
    missing_ids = sum(not record.get("document_id") for record in records)
    identifiers = [record.get("document_id") for record in records]
    if missing_text or missing_ids or len(set(identifiers)) != len(identifiers):
        raise RuntimeError(
            f"Corpus validation failed: {missing_text} missing/empty text records, "
            f"{missing_ids} missing document IDs, "
            f"{len(identifiers) - len(set(identifiers))} duplicate document IDs."
        )


def print_statistics(records: Sequence[Mapping[str, object]]) -> None:
    """Print corpus statistics and deterministic examples for manual inspection."""
    texts = [record.get("text") for record in records]
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("Validated corpus contains invalid text.")
    text_lengths = [len(text) for text in texts if isinstance(text, str)]
    print("\nCorpus verification:")
    print(f"Format: JSON Lines (UTF-8), one AG News article per line")
    print(f"Documents: {len(records):,}")
    print(f"Missing/empty text: 0")
    print(f"Text length (characters): min={min(text_lengths):,}, max={max(text_lengths):,}, average={sum(text_lengths) / len(text_lengths):,.1f}")

    sample_records = random.Random(RANDOM_SEED).sample(records, SAMPLE_COUNT)
    print(f"\nRandom examples (seed={RANDOM_SEED}):")
    for record in sample_records:
        text = record["text"]
        if not isinstance(text, str):
            raise ValueError("Validated corpus contains invalid text.")
        preview = " ".join(text.split())[:180]
        print(f"- {record['document_id']}: {preview}")


def main() -> int:
    """Acquire, validate, and summarize the raw corpus."""
    args = parse_args()
    if args.count <= 0:
        print("Acquisition failed: count must be greater than zero.", file=sys.stderr)
        return 1
    output_path = args.output or default_output_path(args.count)
    output_path = output_path.resolve()

    if output_path.exists():
        print(f"Found existing corpus at {output_path}")
        records = read_records(output_path)
    else:
        try:
            dataset = load_articles()
            write_corpus(dataset, output_path, args.count)
            records = read_records(output_path)
        except Exception as error:
            print(f"Acquisition failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1

    try:
        validate_corpus(records, args.count)
        print_statistics(records)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as error:
        print(f"Corpus validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"\nRaw corpus ready: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
