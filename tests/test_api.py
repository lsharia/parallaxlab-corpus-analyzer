"""API endpoint tests with a mocked RAG pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_rag_pipeline
from src.rag_pipeline import RAGPipeline, RAGResult


class FakePipeline:
    def __init__(
        self,
        error: Exception | None = None,
        result: RAGResult | None = None,
    ) -> None:
        self.error = error
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def answer_query(
        self,
        query: str,
        top_k: int,
        *,
        topic_id: int | None,
        sentiment_label: str | None,
    ) -> RAGResult:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "topic_id": topic_id,
                "sentiment_label": sentiment_label,
            }
        )
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        chunk = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source": "news",
            "text": "Retrieved context.",
            "distance": 0.12,
        }
        return RAGResult(
            query=query,
            answer="Grounded answer.",
            retrieved_chunks=[chunk],
            retrieval_latency_seconds=0.01,
            generation_latency_seconds=0.02,
            total_latency_seconds=0.03,
            context_chunks=[chunk],
            failure_kind=None,
        )


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    app.state.rag_pipeline = None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.rag_pipeline = None


def test_health_does_not_initialize_pipeline(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert app.state.rag_pipeline is None


def test_query_returns_existing_rag_result_shape(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="src.api")
    pipeline = FakePipeline()
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/query", json={"query": "  What happened?  "})

    assert response.status_code == 200
    assert response.json() == {
        "query": "What happened?",
        "answer": "Grounded answer.",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source": "news",
                "text": "Retrieved context.",
                "distance": 0.12,
            }
        ],
        "retrieval_latency_seconds": 0.01,
        "generation_latency_seconds": 0.02,
        "total_latency_seconds": 0.03,
        "context_chunks": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "source": "news",
                "text": "Retrieved context.",
                "distance": 0.12,
            }
        ],
        "failure_kind": None,
    }
    assert pipeline.calls == [
        {
            "query": "What happened?",
            "top_k": 5,
            "topic_id": None,
            "sentiment_label": None,
        }
    ]
    assert "endpoint=/query" in caplog.text
    assert "outcome=success" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "What happened?" not in caplog.text
    assert "Retrieved context." not in caplog.text
    assert "Grounded answer." not in caplog.text


def test_query_passes_valid_optional_filters(client: TestClient) -> None:
    pipeline = FakePipeline()
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post(
        "/query",
        json={
            "query": "Filtered question",
            "top_k": 3,
            "topic_id": 8,
            "sentiment_label": "positive",
        },
    )

    assert response.status_code == 200
    assert pipeline.calls == [
        {
            "query": "Filtered question",
            "top_k": 3,
            "topic_id": 8,
            "sentiment_label": "positive",
        }
    ]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "q", "topic_id": -1},
        {"query": "q", "sentiment_label": "mixed"},
        {"query": "q", "top_k": 0},
        {"query": "q", "unexpected": "field"},
    ],
)
def test_invalid_request_bodies_return_generic_validation_error(
    client: TestClient,
    body: dict[str, Any],
) -> None:
    response = client.post("/query", json=body)

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request body."}


def test_pipeline_exception_is_not_exposed(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="src.api")
    pipeline = FakePipeline(
        RuntimeError("PRIVATE_ERROR PROMPT_CANARY DEEPSEEK_API_KEY=NOT_A_REAL_KEY")
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/query", json={"query": "Question"})

    assert response.status_code == 500
    assert response.json() == {"detail": "The query could not be completed."}
    for private_value in ("PRIVATE_ERROR", "PROMPT_CANARY", "DEEPSEEK_API_KEY"):
        assert private_value not in response.text
        assert private_value not in caplog.text


@pytest.mark.parametrize(
    ("failure_kind", "status_code"),
    [
        ("vector_database_missing", 503),
        ("collection_missing", 503),
        ("missing_api_key", 503),
        ("retrieval_error", 503),
        ("rate_limited", 503),
        ("timeout", 504),
        ("authentication_error", 502),
        ("connection_error", 502),
        ("invalid_response", 502),
        ("output_limit", 502),
        ("api_error", 502),
        ("generation_error", 502),
        ("future_unknown_failure", 500),
    ],
)
def test_expected_pipeline_failures_use_meaningful_status_codes(
    client: TestClient,
    failure_kind: str,
    status_code: int,
) -> None:
    result = RAGResult(
        query="question",
        answer="Safe pipeline message.",
        retrieved_chunks=[],
        retrieval_latency_seconds=0.01,
        generation_latency_seconds=0.0,
        total_latency_seconds=0.01,
        context_chunks=[],
        failure_kind=failure_kind,
    )
    pipeline = FakePipeline(result=result)
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/query", json={"query": "question"})

    assert response.status_code == status_code
    assert response.json()["failure_kind"] == failure_kind
    assert response.json()["answer"] == "Safe pipeline message."


def test_unexpected_error_trace_is_server_logged_without_error_message(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="src.api")
    pipeline = FakePipeline(RuntimeError("PRIVATE_QUERY_AND_SECRET"))
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/query", json={"query": "Question text"})

    assert response.status_code == 500
    assert response.json() == {"detail": "The query could not be completed."}
    assert "PRIVATE_QUERY_AND_SECRET" not in response.text
    assert "PRIVATE_QUERY_AND_SECRET" not in caplog.text
    assert "unexpected_rag_query_error" in caplog.text
    assert "exception_type=RuntimeError" in caplog.text


def test_pipeline_initialization_failure_returns_generic_503(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.ERROR, logger="src.api")

    def fail_initialization():
        raise RuntimeError("PRIVATE_INITIALIZATION_DETAIL")

    monkeypatch.setattr("src.rag_pipeline.RAGPipeline", fail_initialization)

    response = client.post("/query", json={"query": "Question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "The RAG pipeline is unavailable."}
    assert "PRIVATE_INITIALIZATION_DETAIL" not in response.text
    assert "PRIVATE_INITIALIZATION_DETAIL" not in caplog.text
    assert "rag_pipeline_initialization_failed" in caplog.text


@pytest.mark.parametrize(
    ("failure_stage", "failure_kind", "status_code"),
    [("retrieval", "retrieval_error", 503), ("generation", "generation_error", 502)],
)
def test_internal_pipeline_errors_are_logged_once_and_mapped_safely(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
    failure_kind: str,
    status_code: int,
) -> None:
    caplog.set_level(logging.ERROR, logger="src.rag_pipeline")

    class ControlledRetriever:
        def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
            if failure_stage == "retrieval":
                raise RuntimeError("PRIVATE_INTERNAL_QUERY_CONTENT")
            return [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "source": "news",
                    "text": "CONTROLLED_RETRIEVED_DOCUMENT_TEXT",
                    "distance": 0.1,
                }
            ]

    class ControlledLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            assert failure_stage == "generation"
            assert "CONTROLLED_RETRIEVED_DOCUMENT_TEXT" in str(messages)
            raise RuntimeError("PRIVATE_GENERATION_CONFIGURATION")

        def close(self) -> None:
            pass

    pipeline = RAGPipeline(
        retriever=ControlledRetriever(),  # type: ignore[arg-type]
        llm_client=ControlledLLM(),  # type: ignore[arg-type]
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/query", json={"query": "Question"})

    expected_answer = (
        "Answer unavailable: retrieval failed. Please try again."
        if failure_stage == "retrieval"
        else "Answer unavailable: generation failed. Please try again."
    )
    assert response.status_code == status_code
    assert response.json()["failure_kind"] == failure_kind
    assert response.json()["answer"] == expected_answer
    for private_value in (
        "PRIVATE_INTERNAL_QUERY_CONTENT",
        "PRIVATE_GENERATION_CONFIGURATION",
    ):
        assert private_value not in response.text
        assert private_value not in caplog.text
    if failure_stage == "generation":
        assert response.json()["retrieved_chunks"][0]["text"] == (
            "CONTROLLED_RETRIEVED_DOCUMENT_TEXT"
        )
    else:
        assert response.json()["retrieved_chunks"] == []
    assert "CONTROLLED_RETRIEVED_DOCUMENT_TEXT" not in caplog.text
    assert f"operation={failure_stage}" in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert caplog.text.count("rag_pipeline_unexpected_failure") == 1