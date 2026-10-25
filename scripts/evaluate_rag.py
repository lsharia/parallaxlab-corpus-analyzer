"""Evaluate end-to-end RAG responses against manually judged evidence IDs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_retrieval import DEFAULT_DATASET_PATH, validate_dataset
from src.embeddings import DEFAULT_MODEL_NAME
from src.rag_pipeline import INSUFFICIENT_INFORMATION_ANSWER, RAGPipeline
from src.retrieval import SemanticRetriever
from src.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DATABASE_PATH


DEFAULT_OUTPUT_PATH = "data/benchmarks/rag_evaluation.json"
DEFAULT_TOP_K = 5
RETRIEVAL_FAILURE_KINDS = {
    "vector_database_missing",
    "collection_missing",
    "retrieval_error",
}
GENERATION_FAILURE_KINDS = {
    "missing_api_key",
    "authentication_error",
    "rate_limited",
    "timeout",
    "connection_error",
    "invalid_response",
    "output_limit",
    "api_error",
    "generation_error",
    "verification_error",
}


def validate_top_k(top_k: int) -> None:
    """Reject non-positive or non-integer retrieval cutoffs."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _document_ids(chunks: Any) -> list[str]:
    if not isinstance(chunks, list):
        return []
    return list(
        dict.fromkeys(
            chunk["document_id"]
            for chunk in chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get("document_id"), str)
            and chunk["document_id"].strip()
        )
    )


def _latency_statistics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize wall latency for one explicitly selected outcome population."""
    values = [run["end_to_end_latency_seconds"] for run in runs]
    return {
        "run_count": len(values),
        "average_seconds": statistics.mean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "minimum_seconds": min(values) if values else None,
        "maximum_seconds": max(values) if values else None,
    }


def evaluate_case(
    pipeline: Any,
    case: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run one pipeline query and measure evidence availability and latency.

    ``relevant_context_recall_proxy`` measures the fraction of manually judged
    relevant documents present in the pipeline's context chunks. It measures
    evidence availability, not whether the generated answer is semantically
    correct.
    """
    validate_top_k(top_k)
    query = case["query"]
    relevant_document_ids = list(dict.fromkeys(case["relevant_document_ids"]))
    started_at = time.perf_counter()
    try:
        result = pipeline.answer_query(query, top_k=top_k)
        wall_latency = time.perf_counter() - started_at
        retrieved_chunks = result.retrieved_chunks
        context_chunks = result.context_chunks
        failure_kind = result.failure_kind
        answer = result.answer
        retrieved_document_ids = _document_ids(retrieved_chunks)
        context_document_ids = _document_ids(context_chunks)
        relevant_context_ids = sorted(
            set(context_document_ids) & set(relevant_document_ids)
        )
        evidence_recall = (
            len(relevant_context_ids) / len(relevant_document_ids)
            if relevant_document_ids
            else None
        )
        answer_generated = (
            failure_kind is None
            and bool(context_chunks)
            and isinstance(answer, str)
            and bool(answer.strip())
            and answer.strip() != INSUFFICIENT_INFORMATION_ANSWER
        )
        return {
            "query_id": case["query_id"],
            "query": query,
            "relevant_document_ids": relevant_document_ids,
            "retrieved_document_ids": retrieved_document_ids,
            "context_document_ids": context_document_ids,
            "relevant_context_document_ids": relevant_context_ids,
            "relevant_context_recall_proxy": evidence_recall,
            "answer": answer,
            "answer_generated": answer_generated,
            "failure_kind": failure_kind,
            "retrieval_latency_seconds": float(result.retrieval_latency_seconds),
            "generation_latency_seconds": float(result.generation_latency_seconds),
            "verification_latency_seconds": float(
                result.verification_latency_seconds
            ),
            "pipeline_total_latency_seconds": float(result.total_latency_seconds),
            "end_to_end_latency_seconds": wall_latency,
            "grounding_status": result.grounding_status,
        }
    except Exception:
        return {
            "query_id": case["query_id"],
            "query": query,
            "relevant_document_ids": relevant_document_ids,
            "retrieved_document_ids": [],
            "context_document_ids": [],
            "relevant_context_document_ids": [],
            "relevant_context_recall_proxy": (
                0.0 if relevant_document_ids else None
            ),
            "answer": None,
            "answer_generated": False,
            "failure_kind": "unexpected_error",
            "retrieval_latency_seconds": None,
            "generation_latency_seconds": None,
            "verification_latency_seconds": None,
            "pipeline_total_latency_seconds": None,
            "end_to_end_latency_seconds": time.perf_counter() - started_at,
            "grounding_status": None,
        }


def run_evaluation(
    pipeline: Any,
    dataset: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate each labeled query and summarize latency, failures, and evidence."""
    validate_top_k(top_k)
    queries = validate_dataset(dataset)
    runs = [evaluate_case(pipeline, case, top_k) for case in queries]
    failure_counts: dict[str, int] = {}
    for run in runs:
        failure_kind = run["failure_kind"]
        if failure_kind is not None:
            failure_counts[failure_kind] = failure_counts.get(failure_kind, 0) + 1
    retrieval_failure_count = sum(
        count
        for kind, count in failure_counts.items()
        if kind in RETRIEVAL_FAILURE_KINDS
    )
    generation_failure_count = sum(
        count
        for kind, count in failure_counts.items()
        if kind in GENERATION_FAILURE_KINDS
    )
    grounding_verdict_counts: dict[str, int] = {}
    for run in runs:
        verdict = run["grounding_status"]
        if verdict is not None:
            grounding_verdict_counts[verdict] = (
                grounding_verdict_counts.get(verdict, 0) + 1
            )

    scored_evidence = [
        run["relevant_context_recall_proxy"]
        for run in runs
        if run["relevant_context_recall_proxy"] is not None
    ]
    generated_answer_runs = [run for run in runs if run["answer_generated"]]
    failed_runs = [run for run in runs if run["failure_kind"] is not None]
    non_generation_runs = [
        run
        for run in runs
        if not run["answer_generated"] and run["failure_kind"] is None
    ]
    summary = {
        "case_count": len(runs),
        "successful_runs": sum(run["failure_kind"] is None for run in runs),
        "failed_runs": sum(run["failure_kind"] is not None for run in runs),
        "answer_generated_runs": sum(run["answer_generated"] for run in runs),
        "non_generation_runs_without_failure": len(non_generation_runs),
        "failure_counts_by_category": failure_counts,
        "retrieval_failure_count": retrieval_failure_count,
        "generation_failure_count": generation_failure_count,
        "grounding_verdict_counts": grounding_verdict_counts,
        "other_failure_count": sum(
            count
            for kind, count in failure_counts.items()
            if kind not in RETRIEVAL_FAILURE_KINDS | GENERATION_FAILURE_KINDS
        ),
        "latency_seconds": {
            "all_attempts": _latency_statistics(runs),
            "successful_generated_answers": _latency_statistics(
                generated_answer_runs
            ),
            "failed_attempts": _latency_statistics(failed_runs),
        },
        "mean_relevant_context_recall_proxy": (
            statistics.mean(scored_evidence) if scored_evidence else None
        ),
        "evidence_proxy_scored_cases": len(scored_evidence),
    }
    return {
        "dataset_id": dataset["dataset_id"],
        "configuration": {
            "evaluation_unit": "document",
            "top_k": top_k,
            "model_name": (configuration or {}).get("model_name", DEFAULT_MODEL_NAME),
            "collection_name": (configuration or {}).get(
                "collection_name", DEFAULT_COLLECTION_NAME
            ),
            "latency_measurement": "wall time around RAGPipeline.answer_query after pipeline/retriever initialization",
            "answer_quality_measure": "relevant_context_recall_proxy measures evidence availability, not answer correctness; grounding_status is an LLM-assisted check by the same configured model, not an independent factuality guarantee",
            **(configuration or {}),
        },
        "dataset_size": {
            "case_count": len(runs),
            "judged_cases": sum(bool(q["relevant_document_ids"]) for q in queries),
            "unjudged_cases": sum(not q["relevant_document_ids"] for q in queries),
            "relevant_document_judgments": sum(
                len(q["relevant_document_ids"]) for q in queries
            ),
        },
        "summary": summary,
        "runs": runs,
    }


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Write a serialized evaluation report to its requested path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Run the end-to-end RAG evaluation and save its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    dataset_path = (
        args.dataset or PROJECT_ROOT / DEFAULT_DATASET_PATH
    ).resolve()
    output_path = (args.output or PROJECT_ROOT / DEFAULT_OUTPUT_PATH).resolve()
    database_path = (args.database or PROJECT_ROOT / DEFAULT_DATABASE_PATH).resolve()
    pipeline = None
    try:
        validate_top_k(args.top_k)
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        validate_dataset(dataset)
        retriever = SemanticRetriever(
            database_path=database_path,
            collection_name=args.collection,
            model_name=args.model,
            device=args.device,
        )
        pipeline = RAGPipeline(retriever=retriever)
        report = run_evaluation(
            pipeline,
            dataset,
            top_k=args.top_k,
            configuration={
                "database_path": str(database_path),
                "device": args.device,
            },
        )
        save_report(report, output_path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, LookupError, RuntimeError, ValueError) as error:
        print(f"RAG evaluation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if pipeline is not None:
            pipeline.close()

    print(f"Dataset: {report['dataset_id']} ({report['summary']['case_count']} cases)")
    print(f"Successful runs: {report['summary']['successful_runs']}")
    print(f"Failed runs: {report['summary']['failed_runs']}")
    print(f"Failure counts: {report['summary']['failure_counts_by_category']}")
    latency = report["summary"]["latency_seconds"]
    print(f"Attempt latency (all runs): {latency['all_attempts']}")
    print(
        "End-to-end latency (successful generated answers only): "
        f"{latency['successful_generated_answers']}"
    )
    print(f"Attempt latency (failed runs only): {latency['failed_attempts']}")
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())