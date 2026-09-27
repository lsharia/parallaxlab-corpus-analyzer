"""Unit tests for deterministic recursive text chunking."""

from __future__ import annotations

import pandas as pd
import pytest

from src.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document,
    process_dataset,
    split_text,
)


def test_normal_text_splits_into_bounded_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(150))

    chunks = split_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 100 for chunk in chunks)


def test_short_document_stays_as_one_chunk() -> None:
    text = "A short document."

    assert split_text(text, chunk_size=100, chunk_overlap=20) == [text]


def test_empty_text_produces_no_chunks() -> None:
    assert split_text(None) == []
    assert split_text("") == []
    assert split_text("   ") == []


def test_chunk_overlap_is_preserved() -> None:
    chunks = split_text("abcdefghij", chunk_size=6, chunk_overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_chunk_document_preserves_metadata_and_ids() -> None:
    document = {
        "document_id": "doc-123",
        "source": "test-source",
        "text": "This document contains enough text to create several chunks.",
    }

    chunks = chunk_document(document, chunk_size=20, chunk_overlap=5)

    assert chunks
    assert all(chunk["document_id"] == "doc-123" for chunk in chunks)
    assert all(chunk["source"] == "test-source" for chunk in chunks)
    assert [chunk["chunk_id"] for chunk in chunks] == [
        f"doc-123_chunk_{index:04d}" for index in range(len(chunks))
    ]


def test_chunk_ids_are_unique_and_chunks_are_non_empty() -> None:
    documents = [
        {"document_id": "one", "source": "source", "text": "A " * 80},
        {"document_id": "two", "source": "source", "text": "B " * 80},
    ]
    dataframe = pd.DataFrame(documents)

    chunks, _ = process_dataset(dataframe, chunk_size=40, chunk_overlap=5)

    assert chunks["chunk_id"].is_unique
    assert chunks["text"].str.strip().ne("").all()


def test_process_dataset_is_deterministic() -> None:
    dataframe = pd.DataFrame(
        [{"document_id": "doc", "source": "source", "text": "word " * 200}]
    )

    first, first_statistics = process_dataset(
        dataframe, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    )
    second, second_statistics = process_dataset(
        dataframe, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_statistics == second_statistics


def test_chunk_parameters_are_validated() -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("text", chunk_size=10, chunk_overlap=10)
