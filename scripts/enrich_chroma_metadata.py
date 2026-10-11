"""Enrich the existing ChromaDB collection with document-level metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATABASE_PATH,
    DEFAULT_SENTIMENT_METADATA_PATH,
    DEFAULT_TOPIC_METADATA_PATH,
    create_persistent_collection,
    update_collection_metadata,
)


DEFAULT_REPORT_PATH = "data/benchmarks/chroma_metadata_enrichment.json"


def _read_optional_parquet(path: Path) -> tuple[pd.DataFrame | None, bool]:
    if not path.is_file():
        return None, False
    return pd.read_parquet(path), True


def enrich_from_files(
    database_path: Path,
    topic_path: Path,
    sentiment_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Load available metadata outputs and enrich an existing collection."""
    topics, topics_available = _read_optional_parquet(topic_path)
    sentiments, sentiments_available = _read_optional_parquet(sentiment_path)
    _, collection = create_persistent_collection(database_path, collection_name)
    report = update_collection_metadata(
        collection,
        topic_assignments=topics,
        sentiment_scores=sentiments,
        batch_size=batch_size,
    )
    report["collection_name"] = collection.name
    report["missing_inputs"] = [
        str(path)
        for path, available in (
            (topic_path, topics_available),
            (sentiment_path, sentiments_available),
        )
        if not available
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--topics", type=Path, default=None)
    parser.add_argument("--sentiment", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    database_path = (args.database or project_root / DEFAULT_DATABASE_PATH).resolve()
    topic_path = (args.topics or project_root / DEFAULT_TOPIC_METADATA_PATH).resolve()
    sentiment_path = (
        args.sentiment or project_root / DEFAULT_SENTIMENT_METADATA_PATH
    ).resolve()
    report_path = (
        args.report or project_root / DEFAULT_REPORT_PATH
    ).resolve()
    try:
        report = enrich_from_files(
            database_path,
            topic_path,
            sentiment_path,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(
            f"ChromaDB metadata enrichment failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Collection: {report['collection_name']}")
    print(f"Records: {report['records']:,}")
    print(f"Chunks updated: {report['chunks_updated']:,}")
    print(f"Topic coverage: {report['topic_chunks_covered']:,} chunks")
    print(f"Sentiment coverage: {report['sentiment_chunks_covered']:,} chunks")
    print(
        "Unmatched metadata documents: "
        f"topics={report['unmatched_topic_documents']}, "
        f"sentiment={report['unmatched_sentiment_documents']}"
    )
    if report["missing_inputs"]:
        print(f"Missing metadata files (skipped): {', '.join(report['missing_inputs'])}")
    print(f"Coverage report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())