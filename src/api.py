"""FastAPI endpoints for the existing RAG pipeline."""

from __future__ import annotations

import logging
import time
import traceback
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


logger = logging.getLogger(__name__)

RAG_FAILURE_STATUS_CODES = {
    "vector_database_missing": 503,
    "collection_missing": 503,
    "retrieval_error": 503,
    "missing_api_key": 503,
    "rate_limited": 503,
    "timeout": 504,
    "authentication_error": 502,
    "connection_error": 502,
    "invalid_response": 502,
    "output_limit": 502,
    "api_error": 502,
    "generation_error": 502,
    "verification_error": 502,
}


def _log_exception_trace(event: str, error: Exception) -> None:
    """Log traceback locations and exception type, omitting source/error text."""
    frames = "\n".join(
        f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
        for frame in traceback.extract_tb(error.__traceback__)
    )
    logger.error(
        "%s exception_type=%s\n%s",
        event,
        type(error).__name__,
        frames,
    )


class QueryRequest(BaseModel):
    """Validated query and optional retrieval settings."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: Annotated[StrictInt, Field(gt=0)] = 5
    topic_id: Annotated[StrictInt, Field(ge=0)] | None = None
    sentiment_label: Literal["negative", "neutral", "positive"] | None = None

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, value: str) -> str:
        """Normalize the query and reject whitespace-only input."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class RetrievedChunkResponse(BaseModel):
    """One chunk in the existing retriever's result schema."""

    chunk_id: str
    document_id: str | None
    source: str | None
    text: str
    distance: float


class QueryResponse(BaseModel):
    """The answer and diagnostics returned by ``RAGPipeline.answer_query``."""

    model_config = ConfigDict(from_attributes=True)

    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]
    retrieval_latency_seconds: float
    generation_latency_seconds: float
    verification_latency_seconds: float
    total_latency_seconds: float
    context_chunks: list[RetrievedChunkResponse]
    grounding_status: Literal[
        "supported", "unsupported", "insufficient_evidence", "unavailable"
    ] | None
    failure_kind: str | None = None


app = FastAPI(title="RAG Knowledge Extractor API")
app.state.rag_pipeline = None


@app.middleware("http")
async def log_api_request(request: Request, call_next: Any) -> Response:
    """Log one query-free access record with outcome and elapsed time."""
    started_at = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        status_code = response.status_code if response is not None else 500
        outcome = (
            "success"
            if status_code < 400
            else "client_error"
            if status_code < 500
            else "server_error"
        )
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        failure_kind = getattr(request.state, "failure_kind", None)
        level = logging.INFO if status_code < 400 else logging.WARNING
        logger.log(
            level,
            "api_request method=%s endpoint=%s outcome=%s status_code=%d "
            "duration_ms=%.2f failure_kind=%s",
            request.method,
            endpoint,
            outcome,
            status_code,
            elapsed_ms,
            failure_kind or "none",
        )


def get_rag_pipeline(request: Request) -> Any:
    """Create the pipeline on first query, never while importing this module."""
    pipeline = request.app.state.rag_pipeline
    if pipeline is None:
        try:
            from src.rag_pipeline import RAGPipeline

            pipeline = RAGPipeline()
        except Exception as error:
            request.state.failure_kind = "pipeline_unavailable"
            _log_exception_trace(
                "rag_pipeline_initialization_failed endpoint=/query", error
            )
            raise HTTPException(
                status_code=503,
                detail="The RAG pipeline is unavailable.",
            ) from None
        request.app.state.rag_pipeline = pipeline
    return pipeline


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    _request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    """Return a generic validation response without echoing submitted values."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request body."},
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is responding without initializing RAG."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    request: Request,
    response: Response,
    pipeline: Any = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Run one query through the existing RAG pipeline."""
    try:
        result = pipeline.answer_query(
            body.query,
            top_k=body.top_k,
            topic_id=body.topic_id,
            sentiment_label=body.sentiment_label,
        )
        response_model = QueryResponse.model_validate(result)
        request.state.failure_kind = result.failure_kind
        if result.failure_kind is not None:
            response.status_code = RAG_FAILURE_STATUS_CODES.get(
                result.failure_kind, 500
            )
        return response_model
    except Exception as error:
        request.state.failure_kind = "unexpected_error"
        _log_exception_trace("unexpected_rag_query_error endpoint=/query", error)
        raise HTTPException(
            status_code=500,
            detail="The query could not be completed.",
        ) from None