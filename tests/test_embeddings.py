"""Lightweight unit tests for embedding generation without model downloads."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src import embeddings


class FakeSentenceTransformer:
    """Small deterministic model substitute for unit tests."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_sizes: list[int] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.batch_sizes.append(batch_size)
        return np.asarray(
            [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


@pytest.fixture

def fake_model() -> FakeSentenceTransformer:
    return FakeSentenceTransformer("test-model", "cpu")


def test_model_loading_uses_configured_name_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: dict[str, str] = {}

    class FakeModel:
        def __init__(self, model_name: str, device: str) -> None:
            loaded["model_name"] = model_name
            loaded["device"] = device

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(embeddings, "select_device", lambda device: "cpu")

    embeddings.load_embedding_model("custom/test-model", device="auto")

    assert loaded == {"model_name": "custom/test-model", "device": "cpu"}


def test_single_text_embedding_returns_one_vector(fake_model: FakeSentenceTransformer) -> None:
    vectors, metrics = embeddings.embed_texts(["one text"], fake_model, batch_size=4)

    assert vectors.shape == (1, 3)
    assert metrics["number_of_chunks"] == 1
    assert metrics["embedding_dimension"] == 3


def test_multiple_text_embeddings_match_input_count_and_dimension(
    fake_model: FakeSentenceTransformer,
) -> None:
    vectors, metrics = embeddings.embed_texts(
        ["first", "second", "third"], fake_model, batch_size=2
    )

    assert vectors.shape == (3, 3)
    assert metrics["number_of_chunks"] == 3
    assert metrics["embedding_dimension"] == 3
    assert fake_model.batch_sizes == [2]


def test_empty_input_returns_empty_matrix(fake_model: FakeSentenceTransformer) -> None:
    vectors, metrics = embeddings.embed_texts([], fake_model)

    assert vectors.shape == (0, 3)
    assert metrics["number_of_chunks"] == 0
    assert metrics["total_embedding_time"] == 0.0


@pytest.mark.parametrize("texts", [[""], ["  "], ["valid", None]])
def test_invalid_text_inputs_are_rejected(
    texts: list[str | None], fake_model: FakeSentenceTransformer
) -> None:
    with pytest.raises(ValueError, match="empty or non-string"):
        embeddings.embed_texts(texts, fake_model)  # type: ignore[arg-type]


def test_batch_size_must_be_positive(fake_model: FakeSentenceTransformer) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embeddings.embed_texts(["text"], fake_model, batch_size=0)


def test_chunk_id_mapping_is_preserved_and_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = pd.DataFrame(
        {
            "chunk_id": ["chunk-a", "chunk-b"],
            "document_id": ["doc-a", "doc-b"],
            "source": ["source", "source"],
            "text": ["first text", "second text"],
        }
    )
    fake_model = FakeSentenceTransformer("test-model", "cpu")
    monkeypatch.setattr(embeddings, "load_embedding_model", lambda *args: fake_model)
    monkeypatch.setattr(embeddings, "select_device", lambda device: "cpu")

    mapped, metadata = embeddings.generate_chunk_embeddings(
        records, model_name="custom/model", batch_size=1, show_progress_bar=False
    )

    assert mapped["chunk_id"].tolist() == ["chunk-a", "chunk-b"]
    assert mapped["chunk_id"].is_unique
    assert len(mapped["embedding"]) == len(records)
    assert all(len(vector) == 3 for vector in mapped["embedding"])
    assert metadata["model_name"] == "custom/model"
    assert metadata["device"] == "cpu"
    assert metadata["batch_size"] == 1
    assert fake_model.batch_sizes == [1]


def test_non_unique_chunk_ids_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    records = pd.DataFrame(
        {"chunk_id": ["duplicate", "duplicate"], "text": ["one", "two"]}
    )
    monkeypatch.setattr(
        embeddings, "load_embedding_model", lambda *args: pytest.fail("model should not load")
    )

    with pytest.raises(ValueError, match="unique"):
        embeddings.generate_chunk_embeddings(records)
