"""Score cleaned news documents with local VADER sentiment analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DEFAULT_INPUT_PATH = "data/processed/clean_corpus.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/sentiment_scores.parquet"
DEFAULT_LABELS_PATH = "data/labels/manual_sentiment_labels.csv"
DEFAULT_REPORT_PATH = "data/benchmarks/sentiment_evaluation.json"
VADER_VERSION = version("vaderSentiment")
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
SENTIMENT_LABELS = ("negative", "neutral", "positive")
REQUIRED_INPUT_COLUMNS = {"document_id", "text"}
REQUIRED_LABEL_COLUMNS = {"document_id", "sentiment_label"}
REQUIRED_OUTPUT_COLUMNS = {"document_id", "sentiment_label", "sentiment_score"}


def validate_documents(documents: pd.DataFrame) -> None:
    """Validate corpus identity and text columns; null text remains scoreable."""
    if not isinstance(documents, pd.DataFrame):
        raise ValueError("documents must be a pandas DataFrame")
    missing_columns = REQUIRED_INPUT_COLUMNS - set(documents.columns)
    if missing_columns:
        raise ValueError(f"Clean corpus is missing columns: {sorted(missing_columns)}")
    if documents["document_id"].isna().any():
        raise ValueError("Clean corpus contains missing document IDs")
    if not documents["document_id"].is_unique:
        raise ValueError("Clean corpus contains duplicate document IDs")
    invalid_text = documents["text"].map(
        lambda value: not (pd.isna(value) or isinstance(value, str))
    )
    if invalid_text.any():
        raise ValueError("Clean corpus text values must be strings or null")


def validate_sentiment_output(scores: pd.DataFrame) -> None:
    """Validate document sentiment result columns, labels, and score bounds."""
    if not isinstance(scores, pd.DataFrame):
        raise ValueError("scores must be a pandas DataFrame")
    missing_columns = REQUIRED_OUTPUT_COLUMNS - set(scores.columns)
    if missing_columns:
        raise ValueError(f"Sentiment output is missing columns: {sorted(missing_columns)}")
    if scores["document_id"].isna().any() or not scores["document_id"].is_unique:
        raise ValueError("Sentiment document IDs must be present and unique")
    if not scores["sentiment_label"].isin(SENTIMENT_LABELS).all():
        raise ValueError(f"Sentiment labels must be one of: {', '.join(SENTIMENT_LABELS)}")
    try:
        values = scores["sentiment_score"].astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Sentiment scores must be numeric") from error
    if not values.map(math.isfinite).all() or not values.between(-1.0, 1.0).all():
        raise ValueError("Sentiment scores must be finite values between -1 and 1")


def validate_manual_labels(labels: pd.DataFrame) -> None:
    """Validate hand-labeled document IDs and the three supported classes."""
    if not isinstance(labels, pd.DataFrame):
        raise ValueError("labels must be a pandas DataFrame")
    missing_columns = REQUIRED_LABEL_COLUMNS - set(labels.columns)
    if missing_columns:
        raise ValueError(f"Manual labels are missing columns: {sorted(missing_columns)}")
    if labels.empty:
        raise ValueError("Manual labels contain no examples")
    if labels["document_id"].isna().any() or not labels["document_id"].is_unique:
        raise ValueError("Manual-label document IDs must be present and unique")
    if not labels["sentiment_label"].isin(SENTIMENT_LABELS).all():
        raise ValueError(f"Manual labels must be one of: {', '.join(SENTIMENT_LABELS)}")


def score_text(
    text: Any,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> tuple[str, float]:
    """Return VADER's thresholded label and compound score for one document."""
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0

    intensity_analyzer = analyzer or SentimentIntensityAnalyzer()
    compound = float(intensity_analyzer.polarity_scores(text)["compound"])
    if compound >= POSITIVE_THRESHOLD:
        label = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    return label, compound


def score_documents(documents: pd.DataFrame) -> pd.DataFrame:
    """Score every cleaned document while preserving its input order."""
    validate_documents(documents)
    analyzer = SentimentIntensityAnalyzer()
    rows = [score_text(text, analyzer) for text in documents["text"].tolist()]
    scores = pd.DataFrame(
        {
            "document_id": documents["document_id"].reset_index(drop=True),
            "sentiment_label": [row[0] for row in rows],
            "sentiment_score": [row[1] for row in rows],
        }
    )
    validate_sentiment_output(scores)
    return scores


def evaluate_manual_labels(
    documents: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate VADER against independent document-level manual labels."""
    validate_documents(documents)
    validate_manual_labels(labels)
    corpus_by_id = documents.set_index("document_id")
    unknown_ids = set(labels["document_id"]) - set(corpus_by_id.index)
    if unknown_ids:
        raise ValueError(f"Manual labels reference unknown document IDs: {sorted(unknown_ids)[:5]}")

    evaluation_documents = corpus_by_id.loc[labels["document_id"], "text"]
    predictions = [score_text(text) for text in evaluation_documents.tolist()]
    predicted_labels = [prediction[0] for prediction in predictions]
    expected_labels = labels["sentiment_label"].tolist()
    report = classification_report(
        expected_labels,
        predicted_labels,
        labels=list(SENTIMENT_LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "method": "VADER",
        "vader_version": VADER_VERSION,
        "thresholds": {
            "negative_at_or_below": NEGATIVE_THRESHOLD,
            "positive_at_or_above": POSITIVE_THRESHOLD,
        },
        "sample_count": len(labels),
        "class_order": list(SENTIMENT_LABELS),
        "accuracy": float(accuracy_score(expected_labels, predicted_labels)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in SENTIMENT_LABELS
        },
        "confusion_matrix": confusion_matrix(
            expected_labels,
            predicted_labels,
            labels=list(SENTIMENT_LABELS),
        ).tolist(),
    }


def run_sentiment_analysis(
    input_path: Path,
    output_path: Path,
    labels_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Score the corpus, evaluate the manual set, and save generated artifacts."""
    documents = pd.read_parquet(input_path)
    labels = pd.read_csv(labels_path, dtype={"document_id": "string"})
    scores = score_documents(documents)
    evaluation = evaluate_manual_labels(documents, labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "scores_path": output_path,
        "report_path": report_path,
        "document_count": len(scores),
        "evaluation": evaluation,
    }


def main() -> int:
    """Score the clean corpus and save its scores and evaluation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paths = {
        "input_path": (args.input or project_root / DEFAULT_INPUT_PATH).resolve(),
        "output_path": (args.output or project_root / DEFAULT_OUTPUT_PATH).resolve(),
        "labels_path": (args.labels or project_root / DEFAULT_LABELS_PATH).resolve(),
        "report_path": (args.report or project_root / DEFAULT_REPORT_PATH).resolve(),
    }
    try:
        result = run_sentiment_analysis(**paths)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"Sentiment analysis failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    evaluation = result["evaluation"]
    print(f"Scored {result['document_count']:,} documents with VADER {VADER_VERSION}")
    print(f"Scores: {result['scores_path']}")
    print(f"Evaluation report: {result['report_path']}")
    print(f"Manual evaluation samples: {evaluation['sample_count']}")
    print(f"Accuracy: {evaluation['accuracy']:.3f}")
    for label, metrics in evaluation["per_class"].items():
        print(
            f"{label}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
        )
    print(f"Confusion matrix ({', '.join(evaluation['class_order'])}):")
    for row in evaluation["confusion_matrix"]:
        print("  " + " ".join(str(value) for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())