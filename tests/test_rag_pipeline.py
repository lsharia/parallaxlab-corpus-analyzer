"""Unit tests for retrieval-to-answer orchestration without real API calls."""

from __future__ import annotations

from typing import Any

import pytest

from src.llm_client import (
    APIError,
    AuthenticationError,
    InvalidAPIResponseError,
    LLMConnectionError,
    OutputLimitError,
    RateLimitError,
    RequestTimeoutError,
)
from src.rag_pipeline import (
    INSUFFICIENT_INFORMATION_ANSWER,
    RAGPipeline,
)


class FakeRetriever:
    def __init__(self, chunks: Any = None, error: Exception | None = None) -> None:
        self.chunks = [] if chunks is None else chunks
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> Any:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.chunks


class FakeLLMClient:
    def __init__(self, answer: str = "Generated answer.", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.answer

    def close(self) -> None:
        pass


def make_pipeline(
    chunks: Any = None,
    answer: str = "Generated answer.",
    retrieval_error: Exception | None = None,
    llm_error: Exception | None = None,
    prompt_builder: Any = None,
) -> tuple[RAGPipeline, FakeRetriever, FakeLLMClient]:
    retriever = FakeRetriever(chunks, retrieval_error)
    llm_client = FakeLLMClient(answer, llm_error)
    pipeline = RAGPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        llm_client=llm_client,  # type: ignore[arg-type]
        prompt_builder=prompt_builder,
        max_context_chars=256,
    )
    return pipeline, retriever, llm_client


def test_successful_end_to_end_pipeline_preserves_sources_and_latencies() -> None:
    chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source": "guide.md",
            "text": "The guide describes semantic search.",
            "distance": 0.12,
        }
    ]
    captured: dict[str, Any] = {}

    def prompt_builder(query: str, retrieved: list[dict[str, Any]], limit: int):
        captured.update(query=query, retrieved=retrieved, limit=limit)
        return [{"role": "user", "content": "controlled prompt"}]

    pipeline, retriever, llm_client = make_pipeline(
        chunks=chunks,
        answer="The guide describes semantic search.",
        prompt_builder=prompt_builder,
    )

    result = pipeline.answer_query("  What does the guide describe?  ", top_k=3)

    assert result.query == "What does the guide describe?"
    assert result.answer == "The guide describes semantic search."
    assert result.retrieved_chunks == chunks
    assert retriever.calls == [("What does the guide describe?", 3)]
    assert captured == {
        "query": "What does the guide describe?",
        "retrieved": chunks,
        "limit": 256,
    }
    assert llm_client.calls == [[{"role": "user", "content": "controlled prompt"}]]
    assert result.retrieval_latency_seconds >= 0
    assert result.generation_latency_seconds >= 0
    assert result.total_latency_seconds >= 0


def test_empty_retrieval_returns_insufficient_information_without_llm_call() -> None:
    pipeline, _, llm_client = make_pipeline(chunks=[])

    result = pipeline.answer_query("question")

    assert result.answer == INSUFFICIENT_INFORMATION_ANSWER
    assert result.retrieved_chunks == []
    assert result.generation_latency_seconds == 0
    assert llm_client.calls == []


def test_unusable_retrieved_context_is_not_sent_to_llm() -> None:
    pipeline, _, llm_client = make_pipeline(chunks=[{"chunk_id": "missing-text"}])

    result = pipeline.answer_query("question")

    assert result.answer == INSUFFICIENT_INFORMATION_ANSWER
    assert result.retrieved_chunks == [{"chunk_id": "missing-text"}]
    assert llm_client.calls == []


def test_retrieval_failure_returns_a_safe_result() -> None:
    failure = RuntimeError("retrieval unavailable")
    pipeline, _, llm_client = make_pipeline(retrieval_error=failure)

    result = pipeline.answer_query("question")

    assert result.failure_kind == "retrieval_error"
    assert result.answer == "Answer unavailable: retrieval failed. Please try again."
    assert llm_client.calls == []


def test_llm_failure_returns_a_safe_result() -> None:
    failure = RuntimeError("generation unavailable")
    pipeline, _, _ = make_pipeline(
        chunks=[
            {
                "chunk_id": "chunk-1",
                "text": "Some relevant context.",
                "distance": 0.2,
            }
        ],
        llm_error=failure,
    )

    result = pipeline.answer_query("question")

    assert result.failure_kind == "generation_error"
    assert result.answer == "Answer unavailable: generation failed. Please try again."


@pytest.mark.parametrize("query", ["", "   ", None, 42])
def test_invalid_query_is_rejected_before_retrieval(query: Any) -> None:
    pipeline, retriever, llm_client = make_pipeline()

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        pipeline.answer_query(query)  # type: ignore[arg-type]

    assert retriever.calls == []
    assert llm_client.calls == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_is_rejected_before_retrieval(top_k: Any) -> None:
    pipeline, retriever, _ = make_pipeline()

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        pipeline.answer_query("question", top_k=top_k)  # type: ignore[arg-type]

    assert retriever.calls == []


@pytest.mark.parametrize(
    ("error", "failure_kind", "safe_answer"),
    [
        (
            AuthenticationError("private auth response"),
            "authentication_error",
            "rejected its credentials",
        ),
        (RateLimitError("private rate detail"), "rate_limited", "rate limiting"),
        (RequestTimeoutError("private timeout detail"), "timeout", "timed out"),
        (LLMConnectionError("private connection detail"), "connection_error", "connect"),
        (
            InvalidAPIResponseError("private malformed response"),
            "invalid_response",
            "invalid response",
        ),
        (APIError("secret token=private"), "api_error", "could not complete"),
        (OutputLimitError("private token limit"), "output_limit", "provider limit"),
    ],
)
def test_provider_errors_return_safe_user_messages(
    error: Exception, failure_kind: str, safe_answer: str
) -> None:
    pipeline, _, _ = make_pipeline(
        chunks=[
            {
                "chunk_id": "chunk-1",
                "text": "Corpus-supported context.",
                "distance": 0.2,
            }
        ],
        llm_error=error,
    )

    result = pipeline.answer_query("question")

    assert result.failure_kind == failure_kind
    assert safe_answer in result.answer
    assert "private" not in result.answer
    assert "token=" not in result.answer


def test_missing_api_key_returns_safe_result_without_exposing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    retriever = FakeRetriever(
        [{"chunk_id": "chunk-1", "text": "Relevant context.", "distance": 0.2}]
    )
    pipeline = RAGPipeline(retriever=retriever)  # type: ignore[arg-type]

    result = pipeline.answer_query("question")

    assert result.failure_kind == "missing_api_key"
    assert result.answer == "Answer unavailable: the LLM API is not configured."
    assert "DEEPSEEK_API_KEY" not in result.answer


@pytest.mark.parametrize(
    ("retrieval_error", "failure_kind", "safe_answer"),
    [
        (FileNotFoundError("private database path"), "vector_database_missing", "database"),
        (LookupError("private collection detail"), "collection_missing", "collection"),
    ],
)
def test_missing_retrieval_resources_return_safe_results(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_error: Exception,
    failure_kind: str,
    safe_answer: str,
) -> None:
    def fail_to_construct_retriever() -> None:
        raise retrieval_error

    monkeypatch.setattr("src.rag_pipeline.SemanticRetriever", fail_to_construct_retriever)
    pipeline = RAGPipeline(llm_client=FakeLLMClient())  # type: ignore[arg-type]

    result = pipeline.answer_query("question")

    assert result.failure_kind == failure_kind
    assert safe_answer in result.answer
    assert "private" not in result.answer


def test_irrelevant_results_do_not_reach_the_llm() -> None:
    pipeline, _, llm_client = make_pipeline(
        chunks=[
            {
                "chunk_id": "distant",
                "text": "Text unrelated to this question.",
                "distance": 0.94,
            }
        ]
    )

    result = pipeline.answer_query("How do I bake sourdough bread?")

    assert result.answer == INSUFFICIENT_INFORMATION_ANSWER
    assert result.retrieved_chunks[0]["chunk_id"] == "distant"
    assert result.context_chunks == []
    assert llm_client.calls == []


def test_relevance_threshold_is_configurable() -> None:
    chunk = {"chunk_id": "candidate", "text": "Potentially relevant.", "distance": 0.75}
    retriever = FakeRetriever([chunk])
    llm_client = FakeLLMClient()
    pipeline = RAGPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        llm_client=llm_client,  # type: ignore[arg-type]
        max_retrieval_distance=0.8,
    )

    result = pipeline.answer_query("question")

    assert result.answer == "Generated answer."
    assert result.context_chunks == [chunk]
    assert len(llm_client.calls) == 1


def test_invalid_relevance_threshold_is_rejected() -> None:
    for threshold in (-0.1, 2.1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="max_retrieval_distance"):
            RAGPipeline(
                retriever=FakeRetriever(),  # type: ignore[arg-type]
                llm_client=FakeLLMClient(),  # type: ignore[arg-type]
                max_retrieval_distance=threshold,
            )