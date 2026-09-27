"""Lightweight semantic retrieval tests with controlled vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.retrieval import SemanticRetriever
from src.vector_store import create_persistent_collection


class FakeQueryModel:
    """Return fixed vectors and record the sentence-transformer call shape."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": texts,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [self.vectors[text] for text in texts]


def make_store(database_path: Path) -> None:
    """Create a tiny persistent collection with deterministic fake vectors."""
    _, collection = create_persistent_collection(database_path, "rag_chunks")
    collection.upsert(
        ids=["cat-1", "dog-1", "cat-2"],
        documents=[
            "Cats are small domestic animals.",
            "Dogs enjoy playing fetch.",
            "A kitten is a young cat.",
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
        metadatas=[
            {"document_id": "doc-cat", "source": "test"},
            {"document_id": "doc-dog", "source": "test"},
            {"document_id": "doc-kitten", "source": "test"},
        ],
    )


def test_basic_retrieval_returns_expected_fields_and_distance(tmp_path) -> None:
    database_path = tmp_path / "chroma"
    make_store(database_path)
    model = FakeQueryModel({"cats": [1.0, 0.0, 0.0]})
    retriever = SemanticRetriever(database_path=database_path, model=model)

    results = retriever.retrieve("cats", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "cat-1"
    assert set(results[0]) == {"chunk_id", "document_id", "source", "text", "distance"}
    assert results[0]["document_id"] == "doc-cat"
    assert results[0]["source"] == "test"
    assert "cat" in results[0]["text"].lower()
    assert isinstance(results[0]["distance"], float)
    assert retriever.last_latency_seconds >= 0


def test_top_k_limits_results_and_is_clamped_to_collection_size(tmp_path) -> None:
    database_path = tmp_path / "chroma"
    make_store(database_path)
    model = FakeQueryModel({"animals": [1.0, 0.0, 0.0]})
    retriever = SemanticRetriever(database_path=database_path, model=model)

    assert len(retriever.retrieve("animals", top_k=1)) == 1
    assert len(retriever.retrieve("animals", top_k=50)) == 3


@pytest.mark.parametrize("query", ["", "   ", None, 4])
def test_invalid_queries_are_rejected(tmp_path, query: Any) -> None:
    database_path = tmp_path / "chroma"
    make_store(database_path)
    retriever = SemanticRetriever(
        database_path=database_path,
        model=FakeQueryModel({}),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        retriever.retrieve(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_is_rejected(tmp_path, top_k: Any) -> None:
    database_path = tmp_path / "chroma"
    make_store(database_path)
    retriever = SemanticRetriever(
        database_path=database_path,
        model=FakeQueryModel({"q": [1.0, 0.0, 0.0]}),
    )

    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("q", top_k=top_k)  # type: ignore[arg-type]


def test_missing_database_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="database directory does not exist"):
        SemanticRetriever(database_path=tmp_path / "missing", model=FakeQueryModel({}))


def test_missing_collection_is_reported(tmp_path) -> None:
    database_path = tmp_path / "chroma"
    client, _ = create_persistent_collection(database_path, "other_collection")

    with pytest.raises(LookupError, match="does not exist"):
        SemanticRetriever(database_path=database_path, model=FakeQueryModel({}))

    assert client is not None


def test_empty_collection_returns_no_results_without_embedding_query(tmp_path) -> None:
    database_path = tmp_path / "chroma"
    create_persistent_collection(database_path, "rag_chunks")
    model = FakeQueryModel({"query": [1.0, 0.0, 0.0]})
    retriever = SemanticRetriever(database_path=database_path, model=model)

    assert retriever.retrieve("query") == []
    assert model.calls == []


def test_deterministic_order_and_query_embedding_interface(tmp_path) -> None:
    database_path = tmp_path / "chroma"
    make_store(database_path)
    model = FakeQueryModel({"cat query": [1.0, 0.0, 0.0]})
    retriever = SemanticRetriever(database_path=database_path, model=model)

    first = retriever.retrieve("  cat query  ", top_k=3)
    second = retriever.retrieve("cat query", top_k=3)

    assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
    assert model.calls[0] == {
        "texts": ["cat query"],
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert model.calls[1] == model.calls[0]


def test_configured_model_name_is_passed_to_loader(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "chroma"
    make_store(database_path)
    captured: dict[str, Any] = {}
    model = FakeQueryModel({})

    def fake_loader(model_name: str, device: str) -> FakeQueryModel:
        captured.update(model_name=model_name, device=device)
        return model

    monkeypatch.setattr("src.retrieval.load_embedding_model", fake_loader)
    retriever = SemanticRetriever(
        database_path=database_path,
        model_name="custom/model",
        device="cpu",
    )

    assert retriever.model is model
    assert captured == {"model_name": "custom/model", "device": "cpu"}


def test_invalid_configuration_is_rejected(tmp_path) -> None:
    database_path = tmp_path / "chroma"
    make_store(database_path)

    with pytest.raises(ValueError, match="collection_name"):
        SemanticRetriever(database_path=database_path, collection_name=" ", model=object())
    with pytest.raises(ValueError, match="device"):
        SemanticRetriever(database_path=database_path, device="gpu", model=object())
