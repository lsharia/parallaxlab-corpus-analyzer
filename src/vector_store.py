"""Persist chunk documents and precomputed embeddings in ChromaDB."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS_PATH = "data/processed/chunks.parquet"
DEFAULT_EMBEDDINGS_PATH = "data/processed/embeddings.parquet"
DEFAULT_DATABASE_PATH = "data/vector_db"
DEFAULT_COLLECTION_NAME = "rag_chunks"
DEFAULT_BATCH_SIZE = 256


def create_persistent_collection(
    database_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> tuple[Any, Any]:
    """Open or create the persistent ChromaDB client and collection."""
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("The 'chromadb' package is required.") from error

    database_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(database_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _validate_and_align_inputs(chunks: Any, embedding_table: Any) -> list[dict[str, Any]]:
    """Validate both tables and pair every chunk with its vector by chunk ID."""
    required_chunk_columns = {"chunk_id", "document_id", "source", "text"}
    required_embedding_columns = {"chunk_id", "embedding"}
    missing_chunk_columns = required_chunk_columns - set(chunks.columns)
    missing_embedding_columns = required_embedding_columns - set(embedding_table.columns)
    if missing_chunk_columns:
        raise ValueError(f"Chunks missing columns: {sorted(missing_chunk_columns)}")
    if missing_embedding_columns:
        raise ValueError(
            f"Embeddings missing columns: {sorted(missing_embedding_columns)}"
        )
    if chunks.empty or embedding_table.empty:
        raise ValueError("Chunk and embedding datasets must not be empty.")

    chunk_ids = chunks["chunk_id"].tolist()
    embedding_ids = embedding_table["chunk_id"].tolist()
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids):
        raise ValueError("Chunk IDs must be non-empty strings.")
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in embedding_ids):
        raise ValueError("Embedding chunk IDs must be non-empty strings.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk IDs found in chunks dataset.")
    if len(set(embedding_ids)) != len(embedding_ids):
        raise ValueError("Duplicate chunk IDs found in embeddings dataset.")
    if len(chunk_ids) != len(embedding_ids):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunk_ids)} chunks, "
            f"{len(embedding_ids)} embeddings."
        )
    if set(chunk_ids) != set(embedding_ids):
        missing_vectors = set(chunk_ids) - set(embedding_ids)
        extra_vectors = set(embedding_ids) - set(chunk_ids)
        raise ValueError(
            "Chunk/embedding IDs do not match "
            f"(missing vectors: {len(missing_vectors)}, extra vectors: {len(extra_vectors)})."
        )

    vector_by_id = dict(zip(embedding_ids, embedding_table["embedding"].tolist()))
    aligned: list[dict[str, Any]] = []
    expected_dimension: int | None = None
    for record in chunks.to_dict("records"):
        chunk_id = record["chunk_id"]
        text = record["text"]
        document_id = record["document_id"]
        source = record["source"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id!r} has empty or invalid text.")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"Chunk {chunk_id!r} has an invalid document_id.")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Chunk {chunk_id!r} has an invalid source.")

        raw_vector = vector_by_id[chunk_id]
        if raw_vector is None:
            raise ValueError(f"Chunk {chunk_id!r} has no embedding.")
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Chunk {chunk_id!r} has an invalid embedding vector.") from error
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError(f"Chunk {chunk_id!r} has an empty or non-finite embedding.")
        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise ValueError("Embedding vectors must all have the same dimension.")

        aligned.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": source,
                "text": text,
                "embedding": vector,
            }
        )
    return aligned


def ingest_records(
    collection: Any,
    records: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Upsert validated records in bounded batches; return the batch count."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not records:
        raise ValueError("Cannot ingest an empty record list.")

    batch_count = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        collection.upsert(
            ids=[record["chunk_id"] for record in batch],
            documents=[record["text"] for record in batch],
            embeddings=[record["embedding"] for record in batch],
            metadatas=[
                {
                    "document_id": record["document_id"],
                    "source": record["source"],
                }
                for record in batch
            ],
        )
        batch_count += 1
        print(f"Upserted {min(start + len(batch), len(records)):,}/{len(records):,} records")
    return batch_count


def validate_collection(collection: Any, expected_ids: list[str]) -> dict[str, int]:
    """Verify the persisted collection count, IDs, metadata, and embeddings."""
    if not expected_ids:
        raise ValueError("Expected IDs must not be empty.")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected IDs contain duplicates.")

    actual_count = collection.count()
    if actual_count != len(expected_ids):
        raise RuntimeError(
            f"Collection count mismatch: expected {len(expected_ids)}, got {actual_count}."
        )

    stored = collection.get(include=["metadatas", "embeddings"])
    stored_ids = stored["ids"]
    if len(stored_ids) != len(set(stored_ids)):
        raise RuntimeError("Collection returned duplicate IDs.")
    if set(stored_ids) != set(expected_ids):
        raise RuntimeError("Stored collection IDs do not match the input chunk IDs.")

    metadata = stored["metadatas"]
    vectors = stored["embeddings"]
    if metadata is None or len(metadata) != len(expected_ids):
        raise RuntimeError("Collection metadata is missing or incomplete.")
    if vectors is None or len(vectors) != len(expected_ids):
        raise RuntimeError("Collection embeddings are missing or incomplete.")
    if any(
        not item or not item.get("document_id") or not item.get("source")
        for item in metadata
    ):
        raise RuntimeError("One or more records are missing required metadata.")
    if any(vector is None or len(vector) == 0 for vector in vectors):
        raise RuntimeError("One or more stored embeddings are empty.")

    return {"records": actual_count, "unique_ids": len(set(stored_ids))}


def ingest_datasets(
    chunks: Any,
    embedding_table: Any,
    database_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int | str]:
    """Validate input tables, upsert them into ChromaDB, and verify persistence."""
    records = _validate_and_align_inputs(chunks, embedding_table)
    client, collection = create_persistent_collection(database_path, collection_name)
    existing = collection.get(include=["metadatas"])
    existing_ids = set(existing["ids"])
    pending_records = [
        record for record in records if record["chunk_id"] not in existing_ids
    ]
    batches = (
        ingest_records(collection, pending_records, batch_size=batch_size)
        if pending_records
        else 0
    )
    validation = validate_collection(
        collection, [record["chunk_id"] for record in records]
    )
    return {
        "collection_name": collection.name,
        "database_path": str(database_path.resolve()),
        "records": validation["records"],
        "batch_size": batch_size,
        "batches": batches,
    }


def ingest_from_files(
    chunks_path: Path,
    embeddings_path: Path,
    database_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int | str]:
    """Read the existing chunk and embedding Parquet files and ingest them."""
    import pandas as pd

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk dataset does not exist: {chunks_path}")
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding dataset does not exist: {embeddings_path}")

    chunks = pd.read_parquet(chunks_path)
    embedding_table = pd.read_parquet(embeddings_path)
    return ingest_datasets(
        chunks,
        embedding_table,
        database_path,
        collection_name=collection_name,
        batch_size=batch_size,
    )


def main() -> int:
    """Ingest saved chunk vectors into the local persistent ChromaDB store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    chunks_path = (args.chunks or project_root / DEFAULT_CHUNKS_PATH).resolve()
    embeddings_path = (args.embeddings or project_root / DEFAULT_EMBEDDINGS_PATH).resolve()
    database_path = (args.database or project_root / DEFAULT_DATABASE_PATH).resolve()

    try:
        result = ingest_from_files(
            chunks_path,
            embeddings_path,
            database_path,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"Vector ingestion failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print("ChromaDB ingestion validated")
    print(f"Collection: {result['collection_name']}")
    print(f"Database path: {result['database_path']}")
    print(f"Records: {result['records']:,}")
    print(f"Batch size: {result['batch_size']}")
    print(f"Batches: {result['batches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
