"""Unit tests for retrieval benchmark aggregation without real model calls."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.benchmark_retrieval import run_benchmark, validate_configuration


class FakeRetriever:
    """Supply controlled latencies while recording the benchmark call matrix."""

    def __init__(self, latencies: list[float]) -> None:
        self.latencies = iter(latencies)
        self.last_latency_seconds = 0.0
        self.calls: list[tuple[str, int]] = []
        self.model_name = "fake-model"
        self.device = "cpu"
        self.database_path = "temporary-db"
        self.collection_name = "fake-collection"

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self.calls.append((query, top_k))
        self.last_latency_seconds = next(self.latencies)
        return []


def test_benchmark_structure_and_latency_aggregations() -> None:
    queries = ["first topic", "second topic"]
    top_k_values = [1, 3]
    retriever = FakeRetriever([index / 1000 for index in range(1, 9)])

    report = run_benchmark(
        retriever,
        queries=queries,
        top_k_values=top_k_values,
        repetitions=2,
    )

    assert len(report["measurements"]) == 4
    first_measurement = report["measurements"][0]
    assert first_measurement == {
        "query": "first topic",
        "top_k": 1,
        "latencies_ms": [1.0, 2.0],
        "average_latency_ms": 1.5,
        "minimum_latency_ms": 1.0,
        "maximum_latency_ms": 2.0,
    }
    summary = report["summary"]
    assert summary["number_of_queries"] == 2
    assert summary["repetitions"] == 2
    assert summary["number_of_runs"] == 8
    assert summary["average_latency_ms"] == 4.5
    assert summary["median_latency_ms"] == 4.5
    assert summary["minimum_latency_ms"] == 1.0
    assert summary["maximum_latency_ms"] == 8.0
    assert summary["average_latency_by_top_k_ms"] == {"1": 3.5, "3": 5.5}
    assert len(retriever.calls) == 8


def test_empty_benchmark_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="queries"):
        validate_configuration([], [1], 1)
    with pytest.raises(ValueError, match="top_k_values"):
        validate_configuration(["query"], [], 1)


@pytest.mark.parametrize(
    ("queries", "top_k_values", "repetitions", "message"),
    [
        (["  "], [1], 1, "queries"),
        (["query"], [0], 1, "positive integers"),
        (["query"], [1, 1], 1, "duplicates"),
        (["query"], [1], 0, "repetitions"),
    ],
)
def test_invalid_benchmark_configuration_is_rejected(
    queries: list[str],
    top_k_values: list[int],
    repetitions: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_configuration(queries, top_k_values, repetitions)


def test_invalid_retriever_latency_is_rejected() -> None:
    retriever = FakeRetriever([-0.01])

    with pytest.raises(ValueError, match="invalid latency"):
        run_benchmark(retriever, queries=["query"], top_k_values=[1], repetitions=1)
