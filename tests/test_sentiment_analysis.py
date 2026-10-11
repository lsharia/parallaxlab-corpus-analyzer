"""Tests for VADER sentiment scoring and manual evaluation metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sentiment_analysis import (
    evaluate_manual_labels,
    score_documents,
    score_text,
    validate_sentiment_output,
)


def test_sentiment_output_schema_and_score_bounds() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["positive", "negative", "neutral"],
            "text": ["The team won a wonderful victory.", "The company suffered a terrible loss.", "The meeting starts on Tuesday."],
        }
    )

    scores = score_documents(documents)

    assert list(scores.columns) == ["document_id", "sentiment_label", "sentiment_score"]
    assert scores["sentiment_label"].tolist() == ["positive", "negative", "neutral"]
    assert scores["sentiment_score"].between(-1.0, 1.0).all()
    validate_sentiment_output(scores)


@pytest.mark.parametrize("text", [None, float("nan"), "", "   ", "!!!", "2024 123"])
def test_empty_punctuation_and_nonlinguistic_text_is_neutral(text: object) -> None:
    label, score = score_text(text)

    assert label == "neutral"
    assert score == 0.0


def test_short_text_and_ambiguous_text_are_handled() -> None:
    assert score_text("Great!")[0] == "positive"
    assert score_text("Bad.")[0] == "negative"
    assert score_text("The item is on the table.")[0] == "neutral"


def test_technical_jargon_is_scored_without_failure() -> None:
    label, score = score_text("The SP2 kernel patch updates API compatibility.")

    assert label in {"negative", "neutral", "positive"}
    assert -1.0 <= score <= 1.0


def test_batch_scoring_keeps_missing_text_as_neutral() -> None:
    documents = pd.DataFrame(
        {"document_id": ["missing", "blank"], "text": [None, " "]}
    )

    scores = score_documents(documents)

    assert scores["sentiment_label"].tolist() == ["neutral", "neutral"]
    assert scores["sentiment_score"].tolist() == [0.0, 0.0]


def test_manual_evaluation_reports_expected_metrics_and_matrix() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["p1", "p2", "n1", "u1"],
            "text": [
                "A wonderful win for the team.",
                "Excellent results and a great success.",
                "A terrible loss and a horrible defeat.",
                "The committee met on Tuesday.",
            ],
        }
    )
    labels = pd.DataFrame(
        {
            "document_id": ["p1", "p2", "n1", "u1"],
            "sentiment_label": ["positive", "positive", "negative", "neutral"],
        }
    )

    report = evaluate_manual_labels(documents, labels)

    assert report["sample_count"] == 4
    assert report["accuracy"] == 1.0
    assert report["class_order"] == ["negative", "neutral", "positive"]
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert report["per_class"]["positive"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2,
    }


def test_evaluation_rejects_missing_or_unknown_document_ids() -> None:
    documents = pd.DataFrame({"document_id": ["known"], "text": ["A routine report."]})
    missing = pd.DataFrame({"document_id": ["unknown"], "sentiment_label": ["neutral"]})

    with pytest.raises(ValueError, match="unknown document IDs"):
        evaluate_manual_labels(documents, missing)


def test_missing_input_file_is_reported(tmp_path) -> None:
    from src.sentiment_analysis import run_sentiment_analysis

    with pytest.raises(FileNotFoundError):
        run_sentiment_analysis(
            tmp_path / "missing.parquet",
            tmp_path / "scores.parquet",
            tmp_path / "labels.csv",
            tmp_path / "report.json",
        )


def test_output_validation_rejects_invalid_labels_and_scores() -> None:
    invalid_label = pd.DataFrame(
        {"document_id": ["one"], "sentiment_label": ["mixed"], "sentiment_score": [0.0]}
    )
    with pytest.raises(ValueError, match="Sentiment labels"):
        validate_sentiment_output(invalid_label)

    invalid_score = pd.DataFrame(
        {"document_id": ["one"], "sentiment_label": ["neutral"], "sentiment_score": [1.5]}
    )
    with pytest.raises(ValueError, match="between -1 and 1"):
        validate_sentiment_output(invalid_score)