"""Evaluate semantic retrieval against manually judged document IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import DEFAULT_MODEL_NAME
from src.retrieval import SemanticRetriever
from src.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DATABASE_PATH


DEFAULT_DATASET_PATH = "data/labels/retrieval_evaluation_queries.json"
DEFAULT_OUTPUT_PATH = "data/benchmarks/retrieval_evaluation.json"
DEFAULT_K_VALUES = (1, 3, 5, 10)
EVALUATION_UNIT = "document"


def validate_k_values(k_values: Sequence[int]) -> None:
    """Require a nonempty sequence of unique positive integer cutoffs."""
    if isinstance(k_values, (str, bytes)) or not k_values:
        raise ValueError("k_values must not be empty")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in k_values):
        raise ValueError("k_values must contain only positive integers")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must not contain duplicates")


def validate_dataset(dataset: Any) -> list[dict[str, Any]]:
    """Validate the versioned manual-label dataset schema and IDs."""
    if not isinstance(dataset, dict):
        raise ValueError("evaluation dataset must be a JSON object")
    if not isinstance(dataset.get("dataset_id"), str) or not dataset["dataset_id"].strip():
        raise ValueError("evaluation dataset requires a nonempty dataset_id")
    if dataset.get("evaluation_unit") != EVALUATION_UNIT:
        raise ValueError(f"evaluation_unit must be {EVALUATION_UNIT!r}")
    queries = dataset.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("evaluation dataset requires a nonempty queries list")

    seen_query_ids: set[str] = set()
    for index, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(f"query at index {index} must be an object")
        query_id = item.get("query_id")
        query = item.get("query")
        relevant_ids = item.get("relevant_document_ids")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"query at index {index} requires a nonempty query_id")
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_query_ids.add(query_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query {query_id!r} requires nonempty query text")
        if not isinstance(relevant_ids, list):
            raise ValueError(
                f"query {query_id!r} relevant_document_ids must be a list"
            )
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in relevant_ids):
            raise ValueError(
                f"query {query_id!r} contains an invalid relevant document ID"
            )
        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(
                f"query {query_id!r} contains duplicate relevant document IDs"
            )
    return queries


def evaluate_query(
    query: str,
    relevant_document_ids: Sequence[str],
    retrieved_chunks: Sequence[dict[str, Any]],
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Compute document-level Precision@K and Recall@K for one query.

    Results are ranked chunks. The first K chunk positions form the cutoff;
    repeated document IDs consume positions but count as at most one relevant
    hit. Precision always divides by K, including when fewer than K chunks are
    returned. Recall divides by the number of unique manually relevant IDs.
    """
    validate_k_values(k_values)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty")
    if isinstance(relevant_document_ids, (str, bytes)) or any(
        not isinstance(document_id, str) or not document_id.strip()
        for document_id in relevant_document_ids
    ):
        raise ValueError("relevant document IDs must be nonempty strings")
    if len(set(relevant_document_ids)) != len(relevant_document_ids):
        raise ValueError("relevant document IDs must be unique")
    if isinstance(retrieved_chunks, (str, bytes)) or not isinstance(
        retrieved_chunks, Sequence
    ):
        raise ValueError("retrieved_chunks must be a sequence")

    ranked_document_ids: list[str] = []
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        document_id = chunk.get("document_id") if isinstance(chunk, dict) else None
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"retrieved result at rank {rank} has no document_id")
        ranked_document_ids.append(document_id)

    unique_relevant_ids = set(relevant_document_ids)
    if not unique_relevant_ids:
        return {
            "status": "unjudged_no_relevant_labels",
            "relevant_document_count": 0,
            "retrieved_chunk_count": len(retrieved_chunks),
            "retrieved_document_ids": list(dict.fromkeys(ranked_document_ids)),
            "metrics": {
                str(k): {"precision_at_k": None, "recall_at_k": None}
                for k in k_values
            },
        }

    metrics: dict[str, dict[str, float]] = {}
    for k in k_values:
        relevant_hits = set(ranked_document_ids[:k]) & unique_relevant_ids
        metrics[str(k)] = {
            "precision_at_k": len(relevant_hits) / k,
            "recall_at_k": len(relevant_hits) / len(unique_relevant_ids),
        }
    return {
        "status": "scored",
        "relevant_document_count": len(unique_relevant_ids),
        "retrieved_chunk_count": len(retrieved_chunks),
        "retrieved_document_ids": list(dict.fromkeys(ranked_document_ids)),
        "metrics": metrics,
    }


def evaluate_dataset(
    retriever: Any,
    dataset: dict[str, Any],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Run manual queries through a retriever and aggregate macro metrics."""
    validate_k_values(k_values)
    queries = validate_dataset(dataset)
    max_k = max(k_values)
    per_query = []
    for item in queries:
        retrieved_chunks = retriever.retrieve(item["query"], top_k=max_k)
        result = evaluate_query(
            item["query"],
            item["relevant_document_ids"],
            retrieved_chunks,
            k_values,
        )
        per_query.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "relevant_document_ids": item["relevant_document_ids"],
                **result,
            }
        )

    scored_queries = [item for item in per_query if item["status"] == "scored"]
    aggregate = {}
    for k in k_values:
        precision_values = [item["metrics"][str(k)]["precision_at_k"] for item in scored_queries]
        recall_values = [item["metrics"][str(k)]["recall_at_k"] for item in scored_queries]
        aggregate[str(k)] = {
            "macro_precision_at_k": mean(precision_values) if precision_values else None,
            "macro_recall_at_k": mean(recall_values) if recall_values else None,
            "scored_query_count": len(scored_queries),
        }

    return {
        "dataset_id": dataset["dataset_id"],
        "dataset_size": {
            "query_count": len(queries),
            "scored_query_count": len(scored_queries),
            "unjudged_query_count": len(queries) - len(scored_queries),
            "relevant_query_document_judgments": sum(
                len(item["relevant_document_ids"]) for item in queries
            ),
        },
        "configuration": {
            "evaluation_unit": EVALUATION_UNIT,
            "k_values": list(k_values),
            "max_retrieved_chunks_per_query": max_k,
            "precision_denominator": "fixed K; repeated-document chunks consume rank positions",
            "duplicate_document_policy": "deduplicate relevance hits by document_id; a document counts at most once per cutoff",
            "empty_relevance_policy": "no positive labels means metrics are null and the query is excluded from macro averages",
            "aggregate_method": "macro mean across scored queries",
            "model_name": getattr(retriever, "model_name", DEFAULT_MODEL_NAME),
            "device": getattr(retriever, "device", "unknown"),
            "collection_name": getattr(retriever, "collection_name", DEFAULT_COLLECTION_NAME),
        },
        "aggregate_metrics": aggregate,
        "per_query": per_query,
    }


def main() -> int:
    """Run the judged retrieval queries and persist aggregate metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    args = parser.parse_args()

    dataset_path = (args.dataset or PROJECT_ROOT / DEFAULT_DATASET_PATH).resolve()
    output_path = (args.output or PROJECT_ROOT / DEFAULT_OUTPUT_PATH).resolve()
    database_path = (args.database or PROJECT_ROOT / DEFAULT_DATABASE_PATH).resolve()
    try:
        validate_k_values(args.k_values)
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        validate_dataset(dataset)
        retriever = SemanticRetriever(
            database_path=database_path,
            collection_name=args.collection,
            model_name=args.model,
            device=args.device,
        )
        report = evaluate_dataset(retriever, dataset, args.k_values)
        report["configuration"]["database_path"] = str(database_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError, LookupError, RuntimeError, ValueError) as error:
        print(
            f"Retrieval evaluation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Dataset: {dataset['dataset_id']} ({report['dataset_size']['query_count']} queries)")
    print(f"Scored queries: {report['dataset_size']['scored_query_count']}")
    print(f"Unjudged queries: {report['dataset_size']['unjudged_query_count']}")
    for k, metrics in report["aggregate_metrics"].items():
        print(
            f"K={k}: macro Precision@K={metrics['macro_precision_at_k']}, "
            f"macro Recall@K={metrics['macro_recall_at_k']}"
        )
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())