"""Validate the processed clean corpus without rebuilding or downloading it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_COLUMNS = {"document_id", "source", "text"}
MIN_DOCUMENTS = 1


def default_dataset_path() -> Path:
    """Return the expected processed dataset path relative to the project root."""
    return Path(__file__).resolve().parents[1] / "data" / "processed" / "clean_corpus.parquet"


def validate_dataset(dataset_path: Path) -> dict[str, float | int]:
    """Validate the clean Parquet schema, records, IDs, and text statistics."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("The 'pandas' package is required.") from error

    dataframe = pd.read_parquet(dataset_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if len(dataframe) < MIN_DOCUMENTS:
        raise ValueError("The clean dataset contains no documents.")
    if dataframe["document_id"].isna().any():
        raise ValueError("The clean dataset contains missing document IDs.")
    if not dataframe["document_id"].is_unique:
        raise ValueError("The clean dataset contains duplicate document IDs.")

    text = dataframe["text"]
    empty_text = text.isna() | text.astype("string").str.strip().eq("")
    if empty_text.any():
        raise ValueError(f"The clean dataset contains {int(empty_text.sum())} empty texts.")

    text_lengths = text.astype("string").str.len()
    return {
        "documents": len(dataframe),
        "columns": len(dataframe.columns),
        "average_text_length": float(text_lengths.mean()),
        "minimum_text_length": int(text_lengths.min()),
        "maximum_text_length": int(text_lengths.max()),
    }


def main() -> int:
    """Run validation and report basic clean-corpus statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()
    dataset_path = (args.dataset or default_dataset_path()).resolve()

    try:
        statistics = validate_dataset(dataset_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Dataset validation FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print("Dataset validation PASSED")
    print(f"Path: {dataset_path}")
    print(f"Documents: {statistics['documents']:,}")
    print(f"Columns: {statistics['columns']}")
    print(f"Average text length: {statistics['average_text_length']:.1f} characters")
    print(f"Text length range: {statistics['minimum_text_length']}-{statistics['maximum_text_length']} characters")
    print("Required columns present: document_id, source, text")
    print("Empty text values: 0")
    print("Document IDs: unique and non-null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
