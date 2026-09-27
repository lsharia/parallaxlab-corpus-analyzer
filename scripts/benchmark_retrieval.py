"""Benchmark semantic retrieval latency over the local RAG chunk collection."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from src.embeddings import DEFAULT_MODEL_NAME
from src.retrieval import SemanticRetriever
from src.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DATABASE_PATH


DEFAULT_QUERIES = (
    "oil prices and the stock market",
    "mobile phone makers and technology sales",
    "international football and soccer results",
    "company earnings and corporate business",
    "space exploration and science research",
    "world news about international conflicts",
    "politics and national election campaigns",
    "baseball teams and championship sports",
    "movie releases and the entertainment industry",
    "medical research and public health",
)
DEFAULT_TOP_K_VALUES = (1, 3, 5, 10)
DEFAULT_REPETITIONS = 3
DEFAULT_OUTPUT_PATH = "data/benchmarks/retrieval_benchmark.json"


def validate_configuration(
    queries: Sequence[str], top_k_values: Sequence[int], repetitions: int
) -> None:
    """Reject empty or malformed benchmark configurations before running."""
    if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("queries must contain at least one non-empty string")
    if not top_k_values:
        raise ValueError("top_k_values must not be empty")
    if any(
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
        for top_k in top_k_values
    ):
        raise ValueError("top_k_values must contain only positive integers")
    if len(set(top_k_values)) != len(top_k_values):
        raise ValueError("top_k_values must not contain duplicates")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")


def summarize_measurements(
    measurements: list[dict[str, Any]],
    queries: Sequence[str],
    top_k_values: Sequence[int],
    repetitions: int,
) -> dict[str, Any]:
    """Calculate individual-group, overall, and per-top-k latency summaries."""
    validate_configuration(queries, top_k_values, repetitions)
    expected_runs = len(queries) * len(top_k_values) * repetitions
    latencies = [
        latency
        for measurement in measurements
        for latency in measurement["latencies_ms"]
    ]
    if len(measurements) != len(queries) * len(top_k_values):
        raise ValueError("measurement groups do not match the benchmark configuration")
    if len(latencies) != expected_runs:
        raise ValueError("individual latency count does not match repetitions")
    if any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("latencies must be finite non-negative values")

    average_by_top_k = {
        str(top_k): statistics.mean(
            latency
            for measurement in measurements
            if measurement["top_k"] == top_k
            for latency in measurement["latencies_ms"]
        )
        for top_k in top_k_values
    }
    return {
        "number_of_queries": len(queries),
        "repetitions": repetitions,
        "number_of_runs": len(latencies),
        "average_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "minimum_latency_ms": min(latencies),
        "maximum_latency_ms": max(latencies),
        "average_latency_by_top_k_ms": average_by_top_k,
    }


def run_benchmark(
    retriever: Any,
    queries: Sequence[str] = DEFAULT_QUERIES,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Run fixed repeated retrieval calls using an already initialized retriever.

    ``SemanticRetriever.last_latency_seconds`` times only query embedding plus
    the ChromaDB query, excluding model initialization and benchmark setup.
    """
    validate_configuration(queries, top_k_values, repetitions)
    measurements: list[dict[str, Any]] = []

    for query in queries:
        for top_k in top_k_values:
            latencies_ms: list[float] = []
            for _ in range(repetitions):
                retriever.retrieve(query, top_k=top_k)
                latency_seconds = float(retriever.last_latency_seconds)
                if not math.isfinite(latency_seconds) or latency_seconds < 0:
                    raise ValueError("retriever reported an invalid latency")
                latencies_ms.append(latency_seconds * 1_000)
            measurements.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "latencies_ms": latencies_ms,
                    "average_latency_ms": statistics.mean(latencies_ms),
                    "minimum_latency_ms": min(latencies_ms),
                    "maximum_latency_ms": max(latencies_ms),
                }
            )

    summary = summarize_measurements(
        measurements, queries, top_k_values, repetitions
    )
    model = getattr(retriever, "model", None)
    model_device = getattr(model, "device", getattr(retriever, "device", "unknown"))
    return {
        "configuration": {
            "model_name": getattr(retriever, "model_name", DEFAULT_MODEL_NAME),
            "device": str(model_device),
            "database_path": str(getattr(retriever, "database_path", "unknown")),
            "collection_name": getattr(
                retriever, "collection_name", DEFAULT_COLLECTION_NAME
            ),
            "queries": list(queries),
            "top_k_values": list(top_k_values),
            "repetitions": repetitions,
            "timed_operations": ["query embedding", "ChromaDB vector search"],
        },
        "measurements": measurements,
        "summary": summary,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print individual query/top-k measurements and aggregate summaries."""
    for measurement in report["measurements"]:
        latencies = ", ".join(
            f"{latency:.3f}" for latency in measurement["latencies_ms"]
        )
        print(f"Query: {measurement['query']}")
        print(f"top_k={measurement['top_k']}")
        print(f"individual latencies (ms): {latencies}")
        print(
            f"average={measurement['average_latency_ms']:.3f} ms "
            f"min={measurement['minimum_latency_ms']:.3f} ms "
            f"max={measurement['maximum_latency_ms']:.3f} ms\n"
        )

    summary = report["summary"]
    print("Summary:")
    print(f"Queries: {summary['number_of_queries']}")
    print(f"Repetitions: {summary['repetitions']}")
    print(f"Total runs: {summary['number_of_runs']}")
    print(f"Average latency: {summary['average_latency_ms']:.3f} ms")
    print(f"Median latency: {summary['median_latency_ms']:.3f} ms")
    print(f"Minimum latency: {summary['minimum_latency_ms']:.3f} ms")
    print(f"Maximum latency: {summary['maximum_latency_ms']:.3f} ms")
    print("\nLatency by top_k:")
    for top_k, average in summary["average_latency_by_top_k_ms"].items():
        print(f"top_k={top_k}: {average:.3f} ms average")


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Save reproducible benchmark configuration and results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    """Initialize retrieval once, benchmark the configured workload, and save it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--top-k", type=int, nargs="+", default=list(DEFAULT_TOP_K_VALUES))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    database_path = (args.database or project_root / DEFAULT_DATABASE_PATH).resolve()
    output_path = (args.output or project_root / DEFAULT_OUTPUT_PATH).resolve()
    try:
        validate_configuration(DEFAULT_QUERIES, args.top_k, args.repetitions)
        retriever = SemanticRetriever(
            database_path=database_path,
            collection_name=args.collection,
            model_name=args.model,
            device=args.device,
        )
        report = run_benchmark(
            retriever,
            queries=DEFAULT_QUERIES,
            top_k_values=args.top_k,
            repetitions=args.repetitions,
        )
        print(f"Model: {report['configuration']['model_name']}")
        print(f"Device: {report['configuration']['device']}")
        print(f"Database: {database_path}")
        print(f"Collection: {args.collection}")
        print_report(report)
        save_report(report, output_path)
        print(f"\nBenchmark JSON: {output_path}")
    except (FileNotFoundError, LookupError, OSError, RuntimeError, ValueError) as error:
        print(f"Retrieval benchmark failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
