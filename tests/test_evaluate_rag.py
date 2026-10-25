"""Tests for end-to-end RAG evaluation using mocked pipeline results."""

from __future__ import annotations

import pytest

from scripts import evaluate_rag as evaluate_rag_module
from src.rag_pipeline import RAGResult
from scripts.evaluate_rag import evaluate_case, run_evaluation


def make_result(
    query: str,
    *,
    context_document_ids: list[str] | None = None,
    failure_kind: str | None = None,
    answer: str = "Generated answer for manual review.",
    retrieval_latency: float = 0.1,
    generation_latency: float = 0.2,
    verification_latency: float = 0.05,
    total_latency: float = 0.3,
    grounding_status: str | None = "supported",
) -> RAGResult:
    context_document_ids = context_document_ids or []
    context = [
        {
            "chunk_id": f"{document_id}-chunk",
            "document_id": document_id,
            "source": "test",
            "text": f"Evidence from {document_id}",
            "distance": 0.1,
        }
        for document_id in context_document_ids
    ]
    return RAGResult(
        query=query,
        answer=answer,
        retrieved_chunks=context,
        retrieval_latency_seconds=retrieval_latency,
        generation_latency_seconds=generation_latency,
        total_latency_seconds=total_latency,
        context_chunks=context,
        verification_latency_seconds=verification_latency,
        grounding_status=grounding_status,
        failure_kind=failure_kind,
    )


class FakePipeline:
    def __init__(self, results: dict[str, RAGResult | Exception]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def answer_query(self, query: str, top_k: int) -> RAGResult:
        self.calls.append((query, top_k))
        value = self.results[query]
        if isinstance(value, Exception):
            raise value
        return value


def test_evidence_recall_proxy_uses_manual_relevant_document_ids() -> None:
    case = {
        "query_id": "q1",
        "query": "topic question",
        "relevant_document_ids": ["doc-a", "doc-b"],
    }
    pipeline = FakePipeline(
        {
            "topic question": make_result(
                "topic question", context_document_ids=["doc-a"]
            )
        }
    )

    result = evaluate_case(pipeline, case, top_k=4)

    assert result["relevant_context_document_ids"] == ["doc-a"]
    assert result["relevant_context_recall_proxy"] == 0.5
    assert result["answer_generated"] is True
    assert result["answer"] == "Generated answer for manual review."
    assert result["verification_latency_seconds"] == 0.05
    assert result["grounding_status"] == "supported"
    assert pipeline.calls == [("topic question", 4)]


def test_run_evaluation_reports_success_failure_latency_and_macro_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = {
        "dataset_id": "small-test",
        "evaluation_unit": "document",
        "queries": [
            {"query_id": "q1", "query": "first", "relevant_document_ids": ["doc-a", "doc-b"]},
            {"query_id": "q2", "query": "second", "relevant_document_ids": ["doc-c"]},
            {"query_id": "q3", "query": "unjudged", "relevant_document_ids": []},
        ],
    }
    pipeline = FakePipeline(
        {
            "first": make_result("first", context_document_ids=["doc-a"], total_latency=0.3),
            "second": make_result(
                "second",
                context_document_ids=["doc-x"],
                failure_kind="missing_api_key",
                answer="Answer unavailable.",
                total_latency=0.4,
                grounding_status=None,
            ),
            "unjudged": make_result(
                "unjudged",
                context_document_ids=[],
                answer="I don't have enough information.",
                retrieval_latency=0.2,
                generation_latency=0.0,
                total_latency=0.2,
                grounding_status=None,
            ),
        }
    )
    timestamps = iter([0.0, 0.3, 1.0, 1.4, 2.0, 2.2])
    monkeypatch.setattr(
        evaluate_rag_module.time, "perf_counter", lambda: next(timestamps)
    )

    report = run_evaluation(pipeline, dataset, top_k=3)

    summary = report["summary"]
    assert summary["case_count"] == 3
    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 1
    assert summary["answer_generated_runs"] == 1
    assert summary["non_generation_runs_without_failure"] == 1
    assert summary["failure_counts_by_category"] == {"missing_api_key": 1}
    assert summary["retrieval_failure_count"] == 0
    assert summary["generation_failure_count"] == 1
    assert summary["grounding_verdict_counts"] == {"supported": 1}
    assert summary["other_failure_count"] == 0
    assert summary["mean_relevant_context_recall_proxy"] == 0.25
    assert summary["evidence_proxy_scored_cases"] == 2
    latency = summary["latency_seconds"]
    assert latency["all_attempts"]["run_count"] == 3
    assert latency["all_attempts"]["average_seconds"] == pytest.approx(0.3)
    assert latency["successful_generated_answers"]["run_count"] == 1
    assert latency["successful_generated_answers"]["average_seconds"] == pytest.approx(0.3)
    assert latency["failed_attempts"]["run_count"] == 1
    assert latency["failed_attempts"]["average_seconds"] == pytest.approx(0.4)
    assert "average_end_to_end_latency_seconds" not in summary
    assert report["runs"][2]["relevant_context_recall_proxy"] is None
    assert report["runs"][2]["answer_generated"] is False
    assert pipeline.calls == [("first", 3), ("second", 3), ("unjudged", 3)]


def test_unexpected_pipeline_exception_is_counted_without_details() -> None:
    dataset = {
        "dataset_id": "exception-test",
        "evaluation_unit": "document",
        "queries": [
            {"query_id": "q1", "query": "failure", "relevant_document_ids": ["doc-a"]}
        ],
    }
    pipeline = FakePipeline({"failure": RuntimeError("PRIVATE_ERROR_DETAIL")})

    report = run_evaluation(pipeline, dataset)

    run = report["runs"][0]
    assert run["failure_kind"] == "unexpected_error"
    assert run["answer"] is None
    assert report["summary"]["failure_counts_by_category"] == {"unexpected_error": 1}
    assert report["summary"]["other_failure_count"] == 1
    assert "PRIVATE_ERROR_DETAIL" not in str(report)