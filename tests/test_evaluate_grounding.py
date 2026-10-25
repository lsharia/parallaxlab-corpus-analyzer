"""Tests for the offline grounding evaluation and its metric boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_grounding import (
    classification_metrics,
    collect_live_predictions,
    evaluate_protocol_cases,
    evaluate_semantic_cases,
    load_predictions,
    validate_dataset,
)
from src.llm_client import AuthenticationError, LLMClient


def test_protocol_evaluation_measures_parser_and_rejects_invalid_responses() -> None:
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "labels"
        / "grounding_evaluation_cases.json"
    )
    dataset = validate_dataset(json.loads(dataset_path.read_text(encoding="utf-8")))

    result = evaluate_protocol_cases(dataset["protocol_cases"])

    assert result["case_count"] == 8
    assert result["accuracy"] == 1.0
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 1.0
    assert result["per_label"]["error"]["support"] == 5
    assert result["errors"] == []
    assert "not semantic grounding" in result["evaluation_scope"]


def test_semantic_evaluation_is_not_scored_without_observed_predictions() -> None:
    cases = [
        {
            "case_id": "supported",
            "expected_verdict": "supported",
        }
    ]

    result = evaluate_semantic_cases(cases)

    assert result["status"] == "not_scored"
    assert result["precision_recall"] is None


def test_semantic_metrics_score_only_explicitly_supplied_predictions() -> None:
    cases = [
        {"case_id": "supported", "expected_verdict": "supported"},
        {"case_id": "unsupported", "expected_verdict": "unsupported"},
        {"case_id": "insufficient", "expected_verdict": "insufficient_evidence"},
    ]

    result = evaluate_semantic_cases(
        cases,
        {
            "supported": "supported",
            "unsupported": "error",
            "insufficient": "insufficient_evidence",
        },
    )

    assert result["status"] == "scored"
    metrics = result["precision_recall"]
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_label"]["unsupported"]["recall"] == 0
    assert metrics["per_label"]["error"]["precision"] == 0
    assert result["errors"] == [
        {
            "case_id": "unsupported",
            "expected": "unsupported",
            "predicted": "error",
        }
    ]


def test_classification_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty and aligned"):
        classification_metrics(["supported"], [])


def test_load_predictions_accepts_a_saved_case_predictions_report(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "observed.json"
    predictions_path.write_text(
        json.dumps({"case_predictions": {"case-1": "supported"}}),
        encoding="utf-8",
    )

    assert load_predictions(predictions_path) == {"case-1": "supported"}


def test_live_prediction_collection_records_verdicts_and_sanitized_errors() -> None:
    cases = [
        {
            "case_id": "supported",
            "query": "Question?",
            "answer": "Answer.",
            "source_chunks": [{"text": "Evidence."}],
            "expected_verdict": "supported",
            "rationale": "Supported.",
        },
        {
            "case_id": "provider_error",
            "query": "Question?",
            "answer": "Answer.",
            "source_chunks": [{"text": "Evidence."}],
            "expected_verdict": "unsupported",
            "rationale": "Not supported.",
        },
    ]

    class FakeClient(LLMClient):
        def __init__(self) -> None:
            self.responses = iter(
                ['{"verdict":"supported"}', AuthenticationError("private detail")]
            )

        def generate(self, _messages: object) -> str:
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return response

    predictions, errors = collect_live_predictions(cases, FakeClient())

    assert predictions == {"supported": "supported", "provider_error": "error"}
    assert errors == [
        {"case_id": "provider_error", "error_type": "AuthenticationError"}
    ]
    assert "private detail" not in str(errors)
