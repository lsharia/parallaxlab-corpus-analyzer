"""Revalidate topic, sentiment, and filtered retrieval artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import DEFAULT_MODEL_NAME
from src.retrieval import SemanticRetriever
from src.sentiment_analysis import evaluate_manual_labels, score_text
from src.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DATABASE_PATH


DEFAULT_CORPUS_PATH = "data/processed/clean_corpus.parquet"
DEFAULT_TOPIC_ASSIGNMENTS_PATH = "data/processed/topic_assignments.parquet"
DEFAULT_TOPIC_SUMMARY_PATH = "data/processed/topic_summary.parquet"
DEFAULT_SENTIMENT_PATH = "data/processed/sentiment_scores.parquet"
DEFAULT_LABELS_PATH = "data/labels/manual_sentiment_labels.csv"
DEFAULT_REPORT_PATH = "data/benchmarks/nlp_metadata_validation.json"
DEFAULT_TOP_K = 5

TOPIC_ASSESSMENTS = {
    0: (
        "mixed",
        "Politics-heavy terms are mixed with high-probability baseball and archaeology articles.",
    ),
    1: (
        "mixed",
        "Windows and software-security terms recur, but a representative article is political news.",
    ),
    2: (
        "difficult",
        "Music, trials, batteries, and sports terms span representative software, civic, and zoo articles.",
    ),
    3: (
        "mixed",
        "Several representatives cover Iraq and Najaf, while Olympic terms also appear in the word list.",
    ),
    4: (
        "difficult",
        "Oil and stock terms do not align consistently with representative arts, currency, and city articles.",
    ),
    5: (
        "mixed",
        "Sales and profit terms fit some business stories, but representatives also include celebrity news.",
    ),
    6: (
        "difficult",
        "Hurricane terms are not reflected consistently in representative baseball, climate, and soccer stories.",
    ),
    7: (
        "mixed",
        "Athens, court, software, and company terms form a broad news cluster with unrelated policy examples.",
    ),
    8: (
        "coherent",
        "IPO, stock, search, and company terms align broadly with technology and business representatives.",
    ),
    9: (
        "coherent",
        "Spaceflight and science terms align with the China manned-flight and Japanese lunar-probe articles.",
    ),
}


def collect_topic_evidence(
    corpus: pd.DataFrame,
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    documents_per_topic: int = 3,
) -> list[dict[str, Any]]:
    """Return top words and highest-assignment-probability documents per topic."""
    required_summary = {"topic_id", "top_words"}
    required_assignments = {"document_id", "topic_id", "topic_probability"}
    required_corpus = {"document_id", "text"}
    if not required_summary.issubset(summary.columns):
        raise ValueError("Topic summary is missing required columns")
    if not required_assignments.issubset(assignments.columns):
        raise ValueError("Topic assignments are missing required columns")
    if not required_corpus.issubset(corpus.columns):
        raise ValueError("Clean corpus is missing required columns")
    if documents_per_topic <= 0:
        raise ValueError("documents_per_topic must be positive")

    joined = assignments.merge(
        corpus[["document_id", "text"]],
        on="document_id",
        how="left",
        validate="many_to_one",
    )
    evidence = []
    for topic in summary.sort_values("topic_id").itertuples(index=False):
        selected = (
            joined[joined["topic_id"] == topic.topic_id]
            .sort_values("topic_probability", ascending=False)
            .head(documents_per_topic)
        )
        representatives = [
            {
                "document_id": row.document_id,
                "topic_probability": float(row.topic_probability),
                "text_preview": " ".join(str(row.text).split())[:280],
            }
            for row in selected.itertuples(index=False)
        ]
        assessment, observation = TOPIC_ASSESSMENTS.get(
            int(topic.topic_id),
            ("unreviewed", "No manual coherence observation is available for this topic ID."),
        )
        evidence.append(
            {
                "topic_id": int(topic.topic_id),
                "top_words": topic.top_words,
                "assessment": assessment,
                "observation": observation,
                "representative_documents": representatives,
            }
        )
    return evidence


def validate_retrieval_results(
    results: list[dict[str, Any]],
    metadata_by_chunk_id: dict[str, dict[str, Any]],
    topic_id: int | None,
    sentiment_label: str | None,
    top_k: int,
) -> dict[str, Any]:
    """Assert retrieval outputs match requested metadata and preserve schema."""
    expected_result_keys = {"chunk_id", "document_id", "source", "text", "distance"}
    if not results:
        raise RuntimeError("A validation retrieval case returned no results")
    if len(results) > top_k:
        raise RuntimeError(f"Retrieval returned more than top_k={top_k} results")

    checked = []
    for result in results:
        if set(result) != expected_result_keys:
            raise RuntimeError("Retrieval result schema changed")
        metadata = metadata_by_chunk_id.get(result["chunk_id"])
        if metadata is None:
            raise RuntimeError("Retrieved chunk metadata could not be verified")
        if metadata.get("document_id") != result["document_id"]:
            raise RuntimeError("Retrieved document_id does not match stored metadata")
        if topic_id is not None and metadata.get("topic_id") != topic_id:
            raise RuntimeError(f"Retrieved chunk violates topic_id={topic_id} filter")
        if (
            sentiment_label is not None
            and metadata.get("sentiment_label") != sentiment_label
        ):
            raise RuntimeError(
                f"Retrieved chunk violates sentiment_label={sentiment_label!r} filter"
            )
        checked.append(
            {
                "chunk_id": result["chunk_id"],
                "distance": float(result["distance"]),
                "topic_id": metadata.get("topic_id"),
                "sentiment_label": metadata.get("sentiment_label"),
            }
        )
    return {"result_count": len(results), "all_results_match": True, "results": checked}


def validate_filtered_retrieval(
    database_path: Path,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run four real collection queries and verify metadata on every result."""
    retriever = SemanticRetriever(database_path=database_path, collection_name=collection_name)
    cases = [
        {
            "name": "unfiltered",
            "query": "technology companies, stock markets, and recent news",
            "topic_id": None,
            "sentiment_label": None,
        },
        {
            "name": "topic_only",
            "query": "China manned spaceflight and Japanese lunar science",
            "topic_id": 9,
            "sentiment_label": None,
        },
        {
            "name": "sentiment_only",
            "query": "company losses, failures, casualties and falling prices",
            "topic_id": None,
            "sentiment_label": "negative",
        },
        {
            "name": "combined",
            "query": "Google public offering, search company shares and IPO",
            "topic_id": 8,
            "sentiment_label": "positive",
        },
    ]
    validations = []
    for case in cases:
        filters = {
            key: case[key]
            for key in ("topic_id", "sentiment_label")
            if case[key] is not None
        }
        results = retriever.retrieve(case["query"], top_k=top_k, **filters)
        stored = retriever.collection.get(
            ids=[result["chunk_id"] for result in results],
            include=["metadatas"],
        )
        metadata_by_chunk_id = dict(zip(stored["ids"], stored["metadatas"]))
        check = validate_retrieval_results(
            results,
            metadata_by_chunk_id,
            case["topic_id"],
            case["sentiment_label"],
            top_k,
        )
        validations.append(
            {
                "name": case["name"],
                "query": case["query"],
                "filters": filters,
                **check,
            }
        )
    return {
        "collection_name": collection_name,
        "collection_records": retriever.collection.count(),
        "top_k": top_k,
        "embedding_model": DEFAULT_MODEL_NAME,
        "retrieval_cases": validations,
    }


def build_validation_report(
    corpus_path: Path,
    topic_assignments_path: Path,
    topic_summary_path: Path,
    sentiment_path: Path,
    labels_path: Path,
    database_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Build NLP quality, edge-case, manual metric, and retrieval evidence."""
    corpus = pd.read_parquet(corpus_path)
    assignments = pd.read_parquet(topic_assignments_path)
    summary = pd.read_parquet(topic_summary_path)
    sentiments = pd.read_parquet(sentiment_path)
    labels = pd.read_csv(labels_path, dtype={"document_id": "string"})
    evaluation = evaluate_manual_labels(corpus, labels)

    sentiment_cases = [
        ("short", "Win!"),
        ("empty", ""),
        ("jargon", "API returns an HTTP 500 after OAuth token refresh."),
        (
            "named_entities",
            "Microsoft reports a loss while Apple celebrates record profits.",
        ),
        ("ambiguous", "The committee met Tuesday to discuss the proposal."),
    ]
    sentiment_analyzer = score_text
    edge_results = [
        {
            "case": name,
            "text": text,
            "sentiment_label": sentiment_analyzer(text)[0],
            "sentiment_score": sentiment_analyzer(text)[1],
        }
        for name, text in sentiment_cases
    ]
    edge_results.insert(
        1,
        {
            "case": "missing_text",
            "text": None,
            "sentiment_label": score_text(None)[0],
            "sentiment_score": score_text(None)[1],
        },
    )

    retrieval = validate_filtered_retrieval(database_path, collection_name, top_k)
    return {
        "validation_version": 1,
        "corpus_documents": len(corpus),
        "topic_assignments": len(assignments),
        "sentiment_scores": len(sentiments),
        "topic_quality": collect_topic_evidence(corpus, assignments, summary),
        "sentiment_edge_cases": edge_results,
        "sentiment_evaluation": evaluation,
        "filtered_retrieval": retrieval,
        "scope_note": "Retrieval checks validate filter conformance, not improved relevance or answer quality.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--topic-assignments", type=Path, default=None)
    parser.add_argument("--topic-summary", type=Path, default=None)
    parser.add_argument("--sentiment", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = {
        "corpus_path": (args.corpus or PROJECT_ROOT / DEFAULT_CORPUS_PATH).resolve(),
        "topic_assignments_path": (
            args.topic_assignments or PROJECT_ROOT / DEFAULT_TOPIC_ASSIGNMENTS_PATH
        ).resolve(),
        "topic_summary_path": (
            args.topic_summary or PROJECT_ROOT / DEFAULT_TOPIC_SUMMARY_PATH
        ).resolve(),
        "sentiment_path": (args.sentiment or PROJECT_ROOT / DEFAULT_SENTIMENT_PATH).resolve(),
        "labels_path": (args.labels or PROJECT_ROOT / DEFAULT_LABELS_PATH).resolve(),
        "database_path": (args.database or PROJECT_ROOT / DEFAULT_DATABASE_PATH).resolve(),
        "collection_name": args.collection,
        "top_k": args.top_k,
    }
    output_path = (args.output or PROJECT_ROOT / DEFAULT_REPORT_PATH).resolve()
    try:
        report = build_validation_report(**paths)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"NLP metadata validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"Validation report: {output_path}")
    print(f"Topic observations: {len(report['topic_quality'])}")
    print(
        "Sentiment evaluation: "
        f"n={report['sentiment_evaluation']['sample_count']}, "
        f"accuracy={report['sentiment_evaluation']['accuracy']:.3f}"
    )
    for case in report["filtered_retrieval"]["retrieval_cases"]:
        print(
            f"{case['name']}: results={case['result_count']}, "
            f"all_results_match={case['all_results_match']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())