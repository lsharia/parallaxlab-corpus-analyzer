"""Tests for document-level retrieval metrics and report aggregation."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.evaluate_retrieval import (
    evaluate_dataset,
    evaluate_query,
    validate_k_values,
)


def chunk(document_id: str) -> dict[str, str]:
    return {"document_id": document_id}


def test_precision_recall_hand_calculation_and_duplicate_documents() -> None:
    retrieved = [chunk("doc-a"), chunk("doc-a"), chunk("doc-b"), chunk("doc-x")]

    result = evaluate_query(
        "query",
        ["doc-a", "doc-b", "doc-c"],
        retrieved,
        [1, 2, 3, 5],
    )

    assert result["metrics"] == {
        "1": {"precision_at_k": 1.0, "recall_at_k": pytest.approx(1 / 3)},
        "2": {"precision_at_k": 0.5, "recall_at_k": pytest.approx(1 / 3)},
        "3": {"precision_at_k": pytest.approx(2 / 3), "recall_at_k": pytest.approx(2 / 3)},
        "5": {"precision_at_k": 0.4, "recall_at_k": pytest.approx(2 / 3)},
    }
    assert result["retrieved_document_ids"] == ["doc-a", "doc-b", "doc-x"]


def test_empty_results_return_zero_metrics_for_judged_query() -> None:
    result = evaluate_query("query", ["doc-a"], [], [1, 3])

    assert result["metrics"] == {
        "1": {"precision_at_k": 0.0, "recall_at_k": 0.0},
        "3": {"precision_at_k": 0.0, "recall_at_k": 0.0},
    }


def test_no_relevance_labels_are_unjudged_and_have_null_metrics() -> None:
    result = evaluate_query("query", [], [chunk("doc-a")], [1, 5])

    assert result["status"] == "unjudged_no_relevant_labels"
    assert result["metrics"] == {
        "1": {"precision_at_k": None, "recall_at_k": None},
        "5": {"precision_at_k": None, "recall_at_k": None},
    }


@pytest.mark.parametrize("k_values", [[], [0], [-1], [1, 1], [True]])
def test_invalid_k_values_are_rejected(k_values: list[Any]) -> None:
    with pytest.raises(ValueError):
        validate_k_values(k_values)


def test_dataset_macro_averages_and_unjudged_exclusion() -> None:
    class FakeRetriever:
        model_name = "fake-model"
        collection_name = "test-collection"

        def __init__(self) -> None:
            self.results = {
                "first": [chunk("doc-a"), chunk("doc-a"), chunk("doc-b")],
                "second": [chunk("doc-x"), chunk("doc-d"), chunk("doc-d")],
                "unjudged": [chunk("doc-z")],
            }

        def retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
            return self.results[query][:top_k]

    dataset = {
        "dataset_id": "test-set",
        "evaluation_unit": "document",
        "queries": [
            {"query_id": "q1", "query": "first", "relevant_document_ids": ["doc-a", "doc-b"]},
            {"query_id": "q2", "query": "second", "relevant_document_ids": ["doc-d"]},
            {"query_id": "q3", "query": "unjudged", "relevant_document_ids": []},
        ],
    }

    report = evaluate_dataset(FakeRetriever(), dataset, [1, 2, 3])

    assert report["dataset_size"] == {
        "query_count": 3,
        "scored_query_count": 2,
        "unjudged_query_count": 1,
        "relevant_query_document_judgments": 3,
    }
    assert report["aggregate_metrics"] == {
        "1": {"macro_precision_at_k": 0.5, "macro_recall_at_k": 0.25, "scored_query_count": 2},
        "2": {"macro_precision_at_k": 0.5, "macro_recall_at_k": 0.75, "scored_query_count": 2},
        "3": {"macro_precision_at_k": 0.5, "macro_recall_at_k": 1.0, "scored_query_count": 2},
    }
    assert report["per_query"][2]["status"] == "unjudged_no_relevant_labels"


def test_retrieved_chunk_requires_document_id() -> None:
    with pytest.raises(ValueError, match="no document_id"):
        evaluate_query("query", ["doc-a"], [{"chunk_id": "chunk-without-doc"}], [1])