"""Benchmark retrieval-plus-generation latency through the existing RAG pipeline."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from src.embeddings import DEFAULT_MODEL_NAME
from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline
from src.retrieval import SemanticRetriever
from src.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DATABASE_PATH


DEFAULT_QUERIES = (
    "What happened to oil prices and the energy market?",
    "What technology company news affected the stock market?",
    "What international political developments were reported?",
    "What recent results or events involved professional sports teams?",
    "What business or economic news affected consumers?",
)
DEFAULT_TOP_K = 5
DEFAULT_REPETITIONS = 3
DEFAULT_OUTPUT_PATH = "data/benchmarks/rag_benchmark.json"


def validate_configuration(
    queries: Sequence[str], top_k: int, repetitions: int
) -> None:
    """Validate benchmark inputs before initializing models or making API calls."""
    if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("queries must contain at least one non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
    ):
        raise ValueError("repetitions must be a positive integer")


def _latency_seconds(value: Any) -> float:
    latency = float(value)
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("pipeline reported an invalid latency")
    return latency


def benchmark_run(pipeline: Any, query: str, top_k: int) -> dict[str, Any]:
    """Measure one existing pipeline call and return a key-free run record."""
    try:
        result = pipeline.answer_query(query, top_k=top_k)
        retrieval_latency = _latency_seconds(result.retrieval_latency_seconds)
        generation_latency = _latency_seconds(result.generation_latency_seconds)
        failure_kind = result.failure_kind
        context_chunks = getattr(result, "context_chunks", [])
        answer_generated = failure_kind is None and bool(context_chunks)
        return {
            "query": query,
            "top_k": top_k,
            "retrieval_latency_seconds": retrieval_latency,
            "generation_latency_seconds": generation_latency,
            "total_latency_seconds": retrieval_latency + generation_latency,
            "answer_generated": answer_generated,
            "error_type": failure_kind,
        }
    except Exception as error:
        return {
            "query": query,
            "top_k": top_k,
            "retrieval_latency_seconds": None,
            "generation_latency_seconds": None,
            "total_latency_seconds": None,
            "answer_generated": False,
            "error_type": type(error).__name__,
        }


def _summarize_metric(
    runs: Sequence[dict[str, Any]], field: str
) -> tuple[float | None, float | None]:
    values = [run[field] for run in runs if run[field] is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.median(values)


def summarize_runs(
    runs: Sequence[dict[str, Any]], queries: Sequence[str], repetitions: int
) -> dict[str, Any]:
    """Aggregate latency statistics and outcomes, ignoring unmeasured failures."""
    retrieval_average, retrieval_median = _summarize_metric(
        runs, "retrieval_latency_seconds"
    )
    generation_average, generation_median = _summarize_metric(
        runs, "generation_latency_seconds"
    )
    total_average, total_median = _summarize_metric(runs, "total_latency_seconds")
    total_latencies = [
        run["total_latency_seconds"]
        for run in runs
        if run["total_latency_seconds"] is not None
    ]
    failed_runs = sum(run["error_type"] is not None for run in runs)
    successful_runs = sum(run["error_type"] is None for run in runs)
    return {
        "number_of_queries": len(queries),
        "repetitions": repetitions,
        "number_of_runs": len(runs),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "unanswered_runs": sum(not run["answer_generated"] for run in runs),
        "average_retrieval_latency_seconds": retrieval_average,
        "median_retrieval_latency_seconds": retrieval_median,
        "average_generation_latency_seconds": generation_average,
        "median_generation_latency_seconds": generation_median,
        "average_total_latency_seconds": total_average,
        "median_total_latency_seconds": total_median,
        "minimum_total_latency_seconds": min(total_latencies) if total_latencies else None,
        "maximum_total_latency_seconds": max(total_latencies) if total_latencies else None,
    }


def run_benchmark(
    pipeline: Any,
    queries: Sequence[str] = DEFAULT_QUERIES,
    top_k: int = DEFAULT_TOP_K,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Run repeated queries using an already initialized ``RAGPipeline``."""
    validate_configuration(queries, top_k, repetitions)
    runs = [
        benchmark_run(pipeline, query, top_k)
        for query in queries
        for _ in range(repetitions)
    ]
    return {
        "configuration": {
            "queries": list(queries),
            "top_k": top_k,
            "repetitions": repetitions,
            "timed_operations": ["semantic retrieval", "LLM generation"],
            "total_latency_definition": "retrieval_latency_seconds + generation_latency_seconds",
        },
        "runs": runs,
        "summary": summarize_runs(runs, queries, repetitions),
    }


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Write benchmark records and summaries as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _format_latency(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.3f} ms"


def print_report(report: dict[str, Any], output_path: Path) -> None:
    """Print run outcomes and aggregate latency measurements without secrets."""
    for run in report["runs"]:
        print(f"Query: {run['query']}")
        print(f"top_k: {run['top_k']}")
        print(f"retrieval: {_format_latency(run['retrieval_latency_seconds'])}")
        print(f"generation: {_format_latency(run['generation_latency_seconds'])}")
        print(f"total: {_format_latency(run['total_latency_seconds'])}")
        print(f"answer generated: {run['answer_generated']}")
        print(f"error type: {run['error_type'] or 'none'}\n")

    summary = report["summary"]
    print("Summary:")
    print(f"Queries: {summary['number_of_queries']}")
    print(f"Repetitions: {summary['repetitions']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Failed runs: {summary['failed_runs']}")
    print(
        "Average retrieval latency: "
        f"{_format_latency(summary['average_retrieval_latency_seconds'])}"
    )
    print(
        "Median retrieval latency: "
        f"{_format_latency(summary['median_retrieval_latency_seconds'])}"
    )
    print(
        "Average generation latency: "
        f"{_format_latency(summary['average_generation_latency_seconds'])}"
    )
    print(
        "Median generation latency: "
        f"{_format_latency(summary['median_generation_latency_seconds'])}"
    )
    print(f"Average total latency: {_format_latency(summary['average_total_latency_seconds'])}")
    print(f"Median total latency: {_format_latency(summary['median_total_latency_seconds'])}")
    print(f"Minimum total latency: {_format_latency(summary['minimum_total_latency_seconds'])}")
    print(f"Maximum total latency: {_format_latency(summary['maximum_total_latency_seconds'])}")
    print(f"Benchmark JSON: {output_path}")


def main() -> int:
    """Initialize models and storage before timing repeated RAG calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    database_path = (args.database or project_root / DEFAULT_DATABASE_PATH).resolve()
    output_path = (args.output or project_root / DEFAULT_OUTPUT_PATH).resolve()
    try:
        validate_configuration(DEFAULT_QUERIES, args.top_k, args.repetitions)
        llm_client = LLMClient()
        try:
            retriever = SemanticRetriever(
                database_path=database_path,
                collection_name=args.collection,
                model_name=args.model,
                device=args.device,
            )
        except Exception:
            llm_client.close()
            raise

        with RAGPipeline(retriever=retriever, llm_client=llm_client) as pipeline:
            report = run_benchmark(
                pipeline,
                queries=DEFAULT_QUERIES,
                top_k=args.top_k,
                repetitions=args.repetitions,
            )
        save_report(report, output_path)
        print_report(report, output_path)
    except Exception as error:
        if type(error).__name__ == "MissingAPIKeyError":
            print("RAG benchmark unavailable: configure DEEPSEEK_API_KEY.", file=sys.stderr)
        else:
            print(f"RAG benchmark setup failed: {type(error).__name__}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())