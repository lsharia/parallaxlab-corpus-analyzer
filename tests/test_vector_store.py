"""Tests for persistent ChromaDB ingestion using small fake vectors."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.embedding_manifest import chunk_source_fingerprint
from src.vector_store import (
    build_document_metadata,
    create_persistent_collection,
    ingest_datasets,
    ingest_from_files,
    update_collection_metadata,
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


def test_ingestion_updates_changed_records_and_removes_stale_ids(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"
    ingest_datasets(chunks, embeddings, database_path)

    changed_chunks = chunks.iloc[:2].copy()
    changed_embeddings = embeddings.iloc[:2].copy()
    changed_chunks.loc[0, "text"] = "Updated source text"
    changed_embeddings["embedding"] = [
        [0.9, 0.8, 0.7],
        embeddings.iloc[1]["embedding"],
    ]

    result = ingest_datasets(
        changed_chunks,
        changed_embeddings,
        database_path,
        batch_size=2,
    )
    _, collection = create_persistent_collection(database_path)
    stored = collection.get(include=["documents", "embeddings"])
    stored_by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["embeddings"])))

    assert result["records"] == 2
    assert result["batches"] == 1
    assert set(stored_by_id) == {"chunk-1", "chunk-2"}
    assert stored_by_id["chunk-1"][0] == "Updated source text"
    assert list(stored_by_id["chunk-1"][1]) == pytest.approx([0.9, 0.8, 0.7])


def test_explicit_rebuild_allows_embedding_dimension_change(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"
    ingest_datasets(
        chunks,
        embeddings,
        database_path,
        embedding_model_name="model-a",
    )

    changed_dimension = embeddings.copy()
    changed_dimension["embedding"] = [[0.1, 0.2]] * len(changed_dimension)
    with pytest.raises(RuntimeError, match="differs from the collection"):
        ingest_datasets(
            chunks,
            changed_dimension,
            database_path,
            embedding_model_name="model-b",
        )
    result = ingest_datasets(
        chunks,
        changed_dimension,
        database_path,
        rebuild=True,
        embedding_model_name="model-b",
    )
    _, collection = create_persistent_collection(database_path)
    stored = collection.get(include=["embeddings"])

    assert result["records"] == 3
    assert result["batches"] == 1
    assert all(len(vector) == 2 for vector in stored["embeddings"])
    assert collection.metadata["embedding_model"] == "model-b"
    assert collection.metadata["embedding_dimension"] == 2


def test_model_metadata_is_required_to_adopt_existing_collection(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"
    ingest_datasets(chunks, embeddings, database_path)

    with pytest.raises(RuntimeError, match="no recorded embedding configuration"):
        ingest_datasets(
            chunks,
            embeddings,
            database_path,
            embedding_model_name="model-a",
        )

    result = ingest_datasets(
        chunks,
        embeddings,
        database_path,
        rebuild=True,
        embedding_model_name="model-a",
    )
    _, collection = create_persistent_collection(database_path)
    assert result["records"] == 3
    assert collection.metadata["embedding_model"] == "model-a"


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


def test_ingestion_requires_embedding_model_manifest_when_inputs_exist(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    chunks_path = tmp_path / "chunks.parquet"
    embeddings_path = tmp_path / "embeddings.parquet"
    chunks.to_parquet(chunks_path, index=False)
    embeddings.to_parquet(embeddings_path, index=False)

    with pytest.raises(FileNotFoundError, match="Embedding model metadata"):
        ingest_from_files(
            chunks_path,
            embeddings_path,
            tmp_path / "chroma",
        )


def test_file_ingestion_accepts_matching_source_fingerprint(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    chunks_path = tmp_path / "chunks.parquet"
    embeddings_path = tmp_path / "embeddings.parquet"
    manifest_path = tmp_path / "embedding_metadata.json"
    chunks.to_parquet(chunks_path, index=False)
    embeddings.to_parquet(embeddings_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "embedding_dimension": 3,
                "number_of_chunks": len(chunks),
                "chunk_source_sha256": chunk_source_fingerprint(
                    zip(chunks["chunk_id"].tolist(), chunks["text"].tolist())
                ),
            }
        ),
        encoding="utf-8",
    )

    result = ingest_from_files(
        chunks_path,
        embeddings_path,
        tmp_path / "chroma",
        embedding_metadata_path=manifest_path,
    )

    assert result["records"] == len(chunks)
    assert result["embedding_model"] == "test-model"


def test_file_ingestion_rejects_chunks_changed_after_embedding(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    chunks_path = tmp_path / "chunks.parquet"
    embeddings_path = tmp_path / "embeddings.parquet"
    manifest_path = tmp_path / "embedding_metadata.json"
    chunks.to_parquet(chunks_path, index=False)
    embeddings.to_parquet(embeddings_path, index=False)
    original_fingerprint = chunk_source_fingerprint(
        zip(chunks["chunk_id"].tolist(), chunks["text"].tolist())
    )
    manifest_path.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "embedding_dimension": 3,
                "number_of_chunks": len(chunks),
                "chunk_source_sha256": original_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    chunks.loc[0, "text"] = "Changed source text with stale vectors."
    chunks.to_parquet(chunks_path, index=False)

    with pytest.raises(ValueError, match="fingerprint.*regenerate embeddings"):
        ingest_from_files(
            chunks_path,
            embeddings_path,
            tmp_path / "chroma",
            embedding_metadata_path=manifest_path,
        )


def test_document_metadata_is_mapped_and_validated() -> None:
    topics = pd.DataFrame(
        {
            "document_id": ["doc-1", "doc-2"],
            "topic_id": [3, -1],
            "topic_probability": [0.8, 0.0],
        }
    )
    sentiments = pd.DataFrame(
        {
            "document_id": ["doc-1", "doc-2"],
            "sentiment_label": ["positive", "neutral"],
            "sentiment_score": [0.7, 0.0],
        }
    )

    metadata, available = build_document_metadata(topics, sentiments)

    assert metadata == {
        "doc-1": {
            "topic_id": 3,
            "topic_probability": 0.8,
            "sentiment_label": "positive",
            "sentiment_score": 0.7,
        },
        "doc-2": {
            "topic_id": -1,
            "topic_probability": 0.0,
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
        },
    }
    assert available == {
        "topic_documents_available": 1,
        "sentiment_documents_available": 2,
    }


def test_metadata_update_is_idempotent_and_preserves_existing_fields(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"
    ingest_datasets(chunks, embeddings, database_path)
    _, collection = create_persistent_collection(database_path)
    collection.update(
        ids=["chunk-1"],
        metadatas=[
            {
                "document_id": "doc-1",
                "source": "test-source",
                "custom_field": "preserve-me",
            }
        ],
    )
    before = collection.get(include=["documents", "metadatas", "embeddings"])
    topics = pd.DataFrame(
        {
            "document_id": ["doc-1", "doc-unmatched"],
            "topic_id": [2, 1],
            "topic_probability": [0.75, 0.6],
        }
    )
    sentiments = pd.DataFrame(
        {
            "document_id": ["doc-1", "doc-2"],
            "sentiment_label": ["positive", "neutral"],
            "sentiment_score": [0.6, 0.0],
        }
    )

    first = update_collection_metadata(collection, topics, sentiments, batch_size=2)
    after_first = collection.get(include=["documents", "metadatas", "embeddings"])
    second = update_collection_metadata(collection, topics, sentiments, batch_size=2)
    after_second = collection.get(include=["documents", "metadatas", "embeddings"])

    assert first["records"] == second["records"] == 3
    assert first["chunks_updated"] == 3
    assert second["chunks_updated"] == 0
    assert first["topic_documents_matched"] == 1
    assert first["topic_chunks_covered"] == 2
    assert first["sentiment_documents_matched"] == 2
    assert first["sentiment_chunks_covered"] == 3
    assert first["unmatched_topic_documents"] == 1
    assert set(before["ids"]) == set(after_first["ids"]) == set(after_second["ids"])
    assert {row["document_id"] for row in after_first["metadatas"]} == {
        "doc-1",
        "doc-2",
    }
    after_by_id = dict(zip(after_first["ids"], after_first["metadatas"]))
    assert after_by_id["chunk-1"]["source"] == "test-source"
    assert after_by_id["chunk-1"]["custom_field"] == "preserve-me"
    assert after_by_id["chunk-1"]["topic_id"] == 2
    assert after_by_id["chunk-1"]["sentiment_label"] == "positive"
    assert after_by_id["chunk-2"]["topic_id"] == 2
    assert after_by_id["chunk-3"]["sentiment_label"] == "neutral"

    rounded_metadata = dict(after_by_id["chunk-1"])
    rounded_metadata["sentiment_score"] = math.nextafter(
        rounded_metadata["sentiment_score"], math.inf
    )
    collection.update(ids=["chunk-1"], metadatas=[rounded_metadata])
    second = update_collection_metadata(collection, topics, sentiments, batch_size=2)
    after_second = collection.get(include=["documents", "metadatas", "embeddings"])

    assert second["chunks_updated"] == 0

    def normalize(values):
        return [value.tolist() if hasattr(value, "tolist") else value for value in values]

    assert normalize(before["documents"]) == normalize(after_first["documents"])
    assert normalize(before["embeddings"]) == normalize(after_first["embeddings"])
    assert normalize(after_first["documents"]) == normalize(after_second["documents"])
    assert normalize(after_first["embeddings"]) == normalize(after_second["embeddings"])

    unassigned = pd.DataFrame(
        {"document_id": ["doc-1"], "topic_id": [-1], "topic_probability": [0.0]}
    )
    reassigned = update_collection_metadata(collection, topic_assignments=unassigned)
    after_unassigning = collection.get(include=["metadatas"])
    unassigned_by_id = dict(zip(after_unassigning["ids"], after_unassigning["metadatas"]))
    assert reassigned["topic_chunks_covered"] == 0
    assert reassigned["topic_documents_matched"] == 0
    assert unassigned_by_id["chunk-1"]["topic_id"] == -1
    assert unassigned_by_id["chunk-2"]["topic_id"] == -1


def test_metadata_update_handles_missing_tables_and_rejects_bad_rows(tmp_path) -> None:
    chunks, embeddings = make_inputs()
    database_path = tmp_path / "chroma"
    ingest_datasets(chunks, embeddings, database_path)
    _, collection = create_persistent_collection(database_path)

    no_metadata = update_collection_metadata(collection)
    assert no_metadata["chunks_updated"] == 0
    assert no_metadata["chunks_unchanged"] == 3

    duplicate_topics = pd.DataFrame(
        {"document_id": ["doc-1", "doc-1"], "topic_id": [0, 1]}
    )
    with pytest.raises(ValueError, match="duplicate document IDs"):
        update_collection_metadata(collection, topic_assignments=duplicate_topics)

    invalid_sentiments = pd.DataFrame(
        {"document_id": ["doc-1"], "sentiment_label": ["mixed"]}
    )
    with pytest.raises(ValueError, match="invalid sentiment_label"):
        update_collection_metadata(collection, sentiment_scores=invalid_sentiments)

    invalid_topics = pd.DataFrame(
        {"document_id": ["doc-1"], "topic_id": [1.5]}
    )
    with pytest.raises(ValueError, match="invalid topic_id"):
        update_collection_metadata(collection, topic_assignments=invalid_topics)
