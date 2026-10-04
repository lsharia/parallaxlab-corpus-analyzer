"""Orchestrate semantic retrieval, grounded prompt construction, and LLM generation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from src.llm_client import (
    APIError,
    AuthenticationError,
    InvalidAPIResponseError,
    LLMClient,
    LLMConnectionError,
    LLMError,
    MissingAPIKeyError,
    OutputLimitError,
    RateLimitError,
    RequestTimeoutError,
)
from src.prompts import DEFAULT_MAX_CONTEXT_CHARS, build_rag_messages
from src.retrieval import SemanticRetriever


DEFAULT_TOP_K = 5
DEFAULT_MAX_RETRIEVAL_DISTANCE = 0.7
INSUFFICIENT_INFORMATION_ANSWER = "I don't have enough information in the available documents to answer that question."

PromptBuilder = Callable[
    [str, Sequence[Mapping[str, Any]], int], list[dict[str, str]]
]


@dataclass(frozen=True)
class RAGResult:
    """Answer and source/timing data from one RAG query."""

    query: str
    answer: str
    retrieved_chunks: list[Any]
    retrieval_latency_seconds: float
    generation_latency_seconds: float
    total_latency_seconds: float
    context_chunks: list[Any]
    failure_kind: str | None = None


class RAGPipeline:
    """Reusable pipeline joining the existing retriever, prompt builder, and LLM."""

    def __init__(
        self,
        retriever: SemanticRetriever | None = None,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_retrieval_distance: float = DEFAULT_MAX_RETRIEVAL_DISTANCE,
    ) -> None:
        if (
            not isinstance(max_context_chars, int)
            or isinstance(max_context_chars, bool)
            or max_context_chars < 2
        ):
            raise ValueError("max_context_chars must be an integer of at least 2")
        if (
            isinstance(max_retrieval_distance, bool)
            or not isinstance(max_retrieval_distance, (int, float))
            or not math.isfinite(max_retrieval_distance)
            or not 0 <= max_retrieval_distance <= 2
        ):
            raise ValueError("max_retrieval_distance must be between 0 and 2")

        self.retriever = retriever
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or build_rag_messages
        self.max_context_chars = max_context_chars
        self.max_retrieval_distance = float(max_retrieval_distance)

    def answer_query(self, query: str, top_k: int = DEFAULT_TOP_K) -> RAGResult:
        """Retrieve context and generate a grounded answer for ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        normalized_query = query.strip()
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        try:
            if self.retriever is None:
                self.retriever = SemanticRetriever()
            retrieved_chunks = self.retriever.retrieve(normalized_query, top_k=top_k)
            if not isinstance(retrieved_chunks, Sequence) or isinstance(
                retrieved_chunks, (str, bytes)
            ):
                raise TypeError("retriever returned an invalid result")
            chunks = list(retrieved_chunks)
        except FileNotFoundError:
            return self._failure_result(
                normalized_query,
                "Answer unavailable: the vector database is not available.",
                [],
                "vector_database_missing",
                retrieval_started,
                total_started,
            )
        except LookupError:
            return self._failure_result(
                normalized_query,
                "Answer unavailable: the retrieval collection is not available.",
                [],
                "collection_missing",
                retrieval_started,
                total_started,
            )
        except Exception:
            return self._failure_result(
                normalized_query,
                "Answer unavailable: retrieval failed. Please try again.",
                [],
                "retrieval_error",
                retrieval_started,
                total_started,
            )
        retrieval_latency = time.perf_counter() - retrieval_started

        if not chunks:
            return self._result(
                normalized_query,
                INSUFFICIENT_INFORMATION_ANSWER,
                chunks,
                [],
                retrieval_latency,
                0.0,
                total_started,
            )

        context_chunks = [
            chunk
            for chunk in chunks
            if self._is_usable_relevant_chunk(chunk)
        ]
        if not context_chunks:
            return self._result(
                normalized_query,
                INSUFFICIENT_INFORMATION_ANSWER,
                chunks,
                [],
                retrieval_latency,
                0.0,
                total_started,
            )

        try:
            messages = self.prompt_builder(
                normalized_query, context_chunks, self.max_context_chars
            )
        except ValueError:
            return self._result(
                normalized_query,
                INSUFFICIENT_INFORMATION_ANSWER,
                chunks,
                [],
                retrieval_latency,
                0.0,
                total_started,
            )

        generation_started = time.perf_counter()
        try:
            if self.llm_client is None:
                self.llm_client = LLMClient()
            answer = self.llm_client.generate(messages)
        except LLMError as error:
            answer, failure_kind = self._safe_llm_failure(error)
            return self._result(
                normalized_query,
                answer,
                chunks,
                context_chunks,
                retrieval_latency,
                time.perf_counter() - generation_started,
                total_started,
                failure_kind,
            )
        except Exception:
            return self._result(
                normalized_query,
                "Answer unavailable: generation failed. Please try again.",
                chunks,
                context_chunks,
                retrieval_latency,
                time.perf_counter() - generation_started,
                total_started,
                "generation_error",
            )
        generation_latency = time.perf_counter() - generation_started
        return self._result(
            normalized_query,
            answer,
            chunks,
            context_chunks,
            retrieval_latency,
            generation_latency,
            total_started,
        )

    def _is_usable_relevant_chunk(self, chunk: Any) -> bool:
        if not isinstance(chunk, Mapping):
            return False
        text = chunk.get("text")
        distance = chunk.get("distance")
        if not isinstance(text, str) or not text.strip():
            return False
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            return False
        return math.isfinite(distance) and distance <= self.max_retrieval_distance

    @staticmethod
    def _safe_llm_failure(error: LLMError) -> tuple[str, str]:
        if isinstance(error, MissingAPIKeyError):
            return "Answer unavailable: the LLM API is not configured.", "missing_api_key"
        if isinstance(error, AuthenticationError):
            return "Answer unavailable: the LLM provider rejected its credentials.", "authentication_error"
        if isinstance(error, RateLimitError):
            return "Answer unavailable: the LLM provider is rate limiting requests. Try again later.", "rate_limited"
        if isinstance(error, RequestTimeoutError):
            return "Answer unavailable: the LLM request timed out. Please try again.", "timeout"
        if isinstance(error, LLMConnectionError):
            return "Answer unavailable: could not connect to the LLM provider.", "connection_error"
        if isinstance(error, InvalidAPIResponseError):
            return "Answer unavailable: the LLM provider returned an invalid response.", "invalid_response"
        if isinstance(error, OutputLimitError):
            return "Answer unavailable: the request or response exceeded a provider limit.", "output_limit"
        if isinstance(error, APIError):
            return "Answer unavailable: the LLM provider could not complete the request.", "api_error"
        return "Answer unavailable: generation failed. Please try again.", "generation_error"

    @staticmethod
    def _result(
        query: str,
        answer: str,
        retrieved_chunks: list[Any],
        context_chunks: list[Any],
        retrieval_latency: float,
        generation_latency: float,
        total_started: float,
        failure_kind: str | None = None,
    ) -> RAGResult:
        return RAGResult(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            context_chunks=context_chunks,
            retrieval_latency_seconds=retrieval_latency,
            generation_latency_seconds=generation_latency,
            total_latency_seconds=time.perf_counter() - total_started,
            failure_kind=failure_kind,
        )

    @classmethod
    def _failure_result(
        cls,
        query: str,
        answer: str,
        retrieved_chunks: list[Any],
        failure_kind: str,
        retrieval_started: float,
        total_started: float,
    ) -> RAGResult:
        return cls._result(
            query,
            answer,
            retrieved_chunks,
            [],
            time.perf_counter() - retrieval_started,
            0.0,
            total_started,
            failure_kind,
        )

    def close(self) -> None:
        """Close the underlying LLM client's HTTP resources."""
        if self.llm_client is not None:
            self.llm_client.close()

    def __enter__(self) -> RAGPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()