"""Tests for persistent ChromaDB ingestion using small fake vectors."""

from __future__ import annotations

import pandas as pd
import pytest

from src.vector_store import (
    create_persistent_collection,
    ingest_datasets,
    ingest_from_files,
    validate_collection,
)


def make_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two small, deterministic chunk and embedding tables."""
    chunks = pd.DataFrame(
        {
            "chunk_id": ["chunk-1", "chunk-2", "chunk-3"],
            "document_id": ["doc-1", "doc-1", "doc-2"],
            "source": ["test-source", "test-source", "test-source"],
            "text": ["First chunk text", "Second chunk text", "Third chunk text"],
        }
    )
    embeddings = pd.DataFrame(
        {
            "chunk_id": ["chunk-1", "chunk-2", "chunk-3"],
            "embedding": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
        }
    )
    return chunks, embeddings


def test_creates_persistent_collection(tmp_path) -> None:
    _, collection = create_persistent_collection(tmp_path / "chroma")

    assert collection.name == "rag_chunks"
    assert collection.count() == 0


def test_ingests_small_batches_and_preserves_metadata(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"

    result = ingest_datasets(chunks, embeddings, database_path, batch_size=2)
    _, collection = create_persistent_collection(database_path)
    stored = collection.get(include=["documents", "metadatas", "embeddings"])

    assert result["records"] == 3
    assert result["batches"] == 2
    assert collection.count() == 3
    assert set(stored["ids"]) == set(chunks["chunk_id"])
    stored_by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
    assert stored_by_id["chunk-2"] == (
        "Second chunk text",
        {"document_id": "doc-1", "source": "test-source"},
    )
    assert len(stored["embeddings"]) == 3
    assert len(stored["embeddings"][0]) == 3


def test_repeated_ingestion_does_not_duplicate_or_rewrite_records(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"

    first = ingest_datasets(chunks, embeddings, database_path, batch_size=2)
    second = ingest_datasets(chunks, embeddings, database_path, batch_size=2)
    _, collection = create_persistent_collection(database_path)

    assert first["records"] == second["records"] == 3
    assert first["batches"] == 2
    assert second["batches"] == 0
    assert collection.count() == 3


def test_reopens_existing_persistent_database(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"
    ingest_datasets(chunks, embeddings, database_path)

    reopened_client, reopened_collection = create_persistent_collection(database_path)

    assert reopened_client is not None
    assert reopened_collection.count() == 3
    result = validate_collection(reopened_collection, chunks["chunk_id"].tolist())
    assert result == {"records": 3, "unique_ids": 3}


@pytest.mark.parametrize(
    ("chunk_ids", "embedding_ids", "message"),
    [
        (["chunk-1", "chunk-2"], ["chunk-1"], "count mismatch"),
        (["chunk-1", "chunk-2"], ["chunk-1", "other"], "IDs do not match"),
    ],
)
def test_rejects_mismatched_input_counts_or_ids(
    tmp_path, chunk_ids: list[str], embedding_ids: list[str], message: str
) -> None:
    chunks, embeddings = make_inputs()
    chunks = chunks.iloc[: len(chunk_ids)].copy()
    embeddings = embeddings.iloc[: len(embedding_ids)].copy()
    chunks["chunk_id"] = chunk_ids
    embeddings["chunk_id"] = embedding_ids

    with pytest.raises(ValueError, match=message):
        ingest_datasets(chunks, embeddings, tmp_path / "chroma")


def test_rejects_duplicate_ids_before_ingestion(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    chunks.loc[1, "chunk_id"] = "chunk-1"

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        ingest_datasets(chunks, embeddings, tmp_path / "chroma")


def test_rejects_empty_inputs(tmp_path) -> None:
    chunks, embeddings = make_inputs()

    with pytest.raises(ValueError, match="must not be empty"):
        ingest_datasets(chunks.iloc[:0], embeddings.iloc[:0], tmp_path / "chroma")


def test_missing_input_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Chunk dataset does not exist"):
        ingest_from_files(
            tmp_path / "missing_chunks.parquet",
            tmp_path / "missing_embeddings.parquet",
            tmp_path / "chroma",
        )
