"""Mocked tests for RAG latency benchmark records and summaries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmark_rag import benchmark_run, run_benchmark, summarize_runs


class FakePipeline:
    def __init__(self, results: list[Any]) -> None:
        self.results = iter(results)
        self.calls: list[tuple[str, int]] = []

    def answer_query(self, query: str, top_k: int) -> Any:
        self.calls.append((query, top_k))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def make_result(
    retrieval: float,
    generation: float,
    *,
    failure_kind: str | None = None,
    context_chunks: list[dict[str, str]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        retrieval_latency_seconds=retrieval,
        generation_latency_seconds=generation,
        total_latency_seconds=retrieval + generation + 0.01,
        failure_kind=failure_kind,
        context_chunks=[{"text": "context"}] if context_chunks is None else context_chunks,
    )


def test_benchmark_record_uses_stage_sum_for_total_latency() -> None:
    pipeline = FakePipeline([make_result(0.1, 0.3)])

    record = benchmark_run(pipeline, "corpus question", top_k=4)

    assert record["query"] == "corpus question"
    assert record["top_k"] == 4
    assert record["retrieval_latency_seconds"] == 0.1
    assert record["generation_latency_seconds"] == 0.3
    assert record["total_latency_seconds"] == pytest.approx(0.4)
    assert record["answer_generated"] is True
    assert record["error_type"] is None
    assert pipeline.calls == [("corpus question", 4)]


def test_benchmark_records_pipeline_failures_and_keeps_stage_timings() -> None:
    pipeline = FakePipeline([make_result(0.2, 0.05, failure_kind="rate_limited")])

    record = benchmark_run(pipeline, "question", top_k=5)

    assert record["retrieval_latency_seconds"] == 0.2
    assert record["generation_latency_seconds"] == 0.05
    assert record["total_latency_seconds"] == pytest.approx(0.25)
    assert record["answer_generated"] is False
    assert record["error_type"] == "rate_limited"


def test_unexpected_failure_is_recorded_without_exception_text() -> None:
    pipeline = FakePipeline([RuntimeError("secret request details")])

    record = benchmark_run(pipeline, "question", top_k=5)

    assert record["answer_generated"] is False
    assert record["error_type"] == "RuntimeError"
    assert "secret request details" not in str(record)
    assert record["retrieval_latency_seconds"] is None
    assert record["generation_latency_seconds"] is None
    assert record["total_latency_seconds"] is None


def test_run_benchmark_continues_after_individual_failures() -> None:
    pipeline = FakePipeline(
        [
            make_result(0.1, 0.2),
            RuntimeError("private failure"),
            make_result(0.3, 0.4, failure_kind="timeout"),
        ]
    )

    report = run_benchmark(
        pipeline, queries=["first", "second", "third"], top_k=2, repetitions=1
    )

    assert len(report["runs"]) == 3
    assert report["runs"][1]["error_type"] == "RuntimeError"
    assert report["summary"]["successful_runs"] == 1
    assert report["summary"]["failed_runs"] == 2
    assert pipeline.calls == [("first", 2), ("second", 2), ("third", 2)]


def test_summary_statistics_cover_all_latency_components() -> None:
    runs = [
        {
            "retrieval_latency_seconds": 0.1,
            "generation_latency_seconds": 0.2,
            "total_latency_seconds": 0.3,
            "answer_generated": True,
            "error_type": None,
        },
        {
            "retrieval_latency_seconds": 0.3,
            "generation_latency_seconds": 0.4,
            "total_latency_seconds": 0.7,
            "answer_generated": True,
            "error_type": None,
        },
        {
            "retrieval_latency_seconds": None,
            "generation_latency_seconds": None,
            "total_latency_seconds": None,
            "answer_generated": False,
            "error_type": "RuntimeError",
        },
    ]

    summary = summarize_runs(runs, ["first", "second"], repetitions=2)

    assert summary["number_of_queries"] == 2
    assert summary["number_of_runs"] == 3
    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 1
    assert summary["average_retrieval_latency_seconds"] == pytest.approx(0.2)
    assert summary["median_retrieval_latency_seconds"] == pytest.approx(0.2)
    assert summary["average_generation_latency_seconds"] == pytest.approx(0.3)
    assert summary["median_generation_latency_seconds"] == pytest.approx(0.3)
    assert summary["average_total_latency_seconds"] == pytest.approx(0.5)
    assert summary["median_total_latency_seconds"] == pytest.approx(0.5)
    assert summary["minimum_total_latency_seconds"] == 0.3
    assert summary["maximum_total_latency_seconds"] == 0.7


def test_no_context_counts_as_no_answer_without_a_request_error() -> None:
    pipeline = FakePipeline([make_result(0.12, 0.0, context_chunks=[])])

    report = run_benchmark(pipeline, queries=["query"], top_k=5, repetitions=1)

    assert report["runs"][0]["answer_generated"] is False
    assert report["runs"][0]["error_type"] is None
    assert report["summary"]["unanswered_runs"] == 1


@pytest.mark.parametrize(
    ("queries", "top_k", "repetitions", "message"),
    [
        ([], 5, 1, "queries"),
        (["query"], 0, 1, "top_k"),
        (["query"], 1, 0, "repetitions"),
    ],
)
def test_invalid_benchmark_configuration_is_rejected(
    queries: list[str], top_k: int, repetitions: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_benchmark(FakePipeline([]), queries, top_k, repetitions)