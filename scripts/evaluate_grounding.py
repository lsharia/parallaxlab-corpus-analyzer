"""Evaluate grounding protocol fixtures and optional semantic observations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grounding import (
    AnswerGroundingVerifier,
    GroundingVerificationError,
    build_grounding_messages,
)
from src.llm_client import LLMClient, LLMError


DEFAULT_DATASET_PATH: str = "data/labels/grounding_evaluation_cases.json"
DEFAULT_OUTPUT_PATH: str = "data/benchmarks/grounding_evaluation.json"
VERDICT_LABELS: tuple[str, ...] = (
    "supported",
    "unsupported",
    "insufficient_evidence",
    "error",
)


class SemanticCase(TypedDict):
    """Manually labeled source-and-answer example."""

    case_id: str
    query: str
    source_chunks: list[dict[str, str]]
    answer: str
    expected_verdict: Literal["supported", "unsupported", "insufficient_evidence"]
    rationale: str


class ProtocolCase(TypedDict):
    """Fixed response fixture with its manually expected parser output."""

    case_id: str
    response: str
    expected_output: Literal[
        "supported", "unsupported", "insufficient_evidence", "error"
    ]


class EvaluationDataset(TypedDict):
    """Grounding evaluation labels and response fixtures."""

    dataset_id: str
    annotation_note: str
    cases: list[SemanticCase]
    protocol_cases: list[ProtocolCase]


class ClassificationMetrics(TypedDict):
    """Exact-label metrics for the four supported parser outcomes."""

    case_count: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    per_label: dict[str, dict[str, float | int | None]]
    confusion_matrix: dict[str, dict[str, int]]


class ProtocolEvaluation(ClassificationMetrics):
    """Response-contract metrics and mismatching parser cases."""

    evaluation_scope: str
    errors: list[dict[str, str]]


class SemanticEvaluation(TypedDict):
    """Semantic metrics when observed predictions are supplied."""

    status: Literal["not_scored", "scored"]
    case_count: int
    precision_recall: ClassificationMetrics | None
    reason: NotRequired[str]
    evaluation_scope: NotRequired[str]
    errors: NotRequired[list[dict[str, str]]]


class LivePredictionError(TypedDict):
    """Sanitized failure metadata for one observed verifier request."""

    case_id: str
    error_type: str


def validate_dataset(dataset: object) -> EvaluationDataset:
    """Validate the manually labeled semantic and response-contract cases."""
    if not isinstance(dataset, dict):
        raise ValueError("evaluation dataset must be a JSON object")
    if not isinstance(dataset.get("dataset_id"), str) or not dataset["dataset_id"].strip():
        raise ValueError("evaluation dataset requires a nonempty dataset_id")

    semantic_cases = dataset.get("cases")
    if not isinstance(semantic_cases, list) or not semantic_cases:
        raise ValueError("evaluation dataset requires nonempty semantic cases")
    seen_ids: set[str] = set()
    for index, case in enumerate(semantic_cases):
        if not isinstance(case, dict):
            raise ValueError(f"semantic case at index {index} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError(f"semantic case at index {index} has an invalid or duplicate ID")
        seen_ids.add(case_id)
        if (
            not isinstance(case.get("query"), str)
            or not case["query"].strip()
            or not isinstance(case.get("answer"), str)
            or not case["answer"].strip()
        ):
            raise ValueError(f"semantic case {case_id!r} requires query and answer text")
        sources = case.get("source_chunks")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"semantic case {case_id!r} requires source_chunks")
        if case.get("expected_verdict") not in VERDICT_LABELS[:3]:
            raise ValueError(f"semantic case {case_id!r} has an invalid expected_verdict")
        if not isinstance(case.get("rationale"), str) or not case["rationale"].strip():
            raise ValueError(f"semantic case {case_id!r} requires a rationale")
        # Reuse the production serializer's evidence validation.
        build_grounding_messages(case["query"], case["answer"], sources)

    protocol_cases = dataset.get("protocol_cases")
    if not isinstance(protocol_cases, list) or not protocol_cases:
        raise ValueError("evaluation dataset requires nonempty protocol_cases")
    protocol_ids: set[str] = set()
    for index, case in enumerate(protocol_cases):
        if not isinstance(case, dict):
            raise ValueError(f"protocol case at index {index} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in protocol_ids:
            raise ValueError(f"protocol case at index {index} has an invalid or duplicate ID")
        protocol_ids.add(case_id)
        if not isinstance(case.get("response"), str):
            raise ValueError(f"protocol case {case_id!r} requires a string response")
        if case.get("expected_output") not in VERDICT_LABELS:
            raise ValueError(f"protocol case {case_id!r} has an invalid expected_output")
    return cast(EvaluationDataset, dataset)


def classification_metrics(
    expected: list[str],
    predicted: list[str],
) -> ClassificationMetrics:
    """Compute exact-label confusion counts, precision, recall, and accuracy."""
    if len(expected) != len(predicted) or not expected:
        raise ValueError("expected and predicted labels must be nonempty and aligned")
    confusion = {
        label: {
            prediction: sum(
                truth == label and guess == prediction
                for truth, guess in zip(expected, predicted)
            )
            for prediction in VERDICT_LABELS
        }
        for label in VERDICT_LABELS
    }
    per_label: dict[str, dict[str, float | int | None]] = {}
    scored_labels: list[str] = []
    for label in VERDICT_LABELS:
        true_positive = confusion[label][label]
        predicted_count = sum(confusion[truth][label] for truth in VERDICT_LABELS)
        actual_count = sum(confusion[label].values())
        if actual_count:
            scored_labels.append(label)
        per_label[label] = {
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / actual_count if actual_count else None,
            "support": actual_count,
        }
    scored_precisions = [
        float(per_label[label]["precision"]) for label in scored_labels
    ]
    scored_recalls = [
        float(per_label[label]["recall"]) for label in scored_labels
    ]
    return {
        "case_count": len(expected),
        "accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(expected),
        "macro_precision": sum(scored_precisions) / len(scored_precisions),
        "macro_recall": sum(scored_recalls) / len(scored_recalls),
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def evaluate_protocol_cases(cases: list[ProtocolCase]) -> ProtocolEvaluation:
    """Measure parser conformance on fixed valid and invalid response fixtures."""
    expected: list[str] = []
    predicted: list[str] = []
    errors: list[dict[str, str]] = []
    verifier = AnswerGroundingVerifier()
    for case in cases:
        try:
            actual = verifier.verify(
                "What does the source say?",
                "A short candidate answer.",
                [{"chunk_id": "fixture", "source": "fixture", "text": "Fixture source."}],
                lambda _messages, response=case["response"]: response,
            )
        except GroundingVerificationError:
            actual = "error"
        expected_output = case["expected_output"]
        expected.append(expected_output)
        predicted.append(actual)
        if actual != expected_output:
            errors.append(
                {
                    "case_id": case["case_id"],
                    "expected": expected_output,
                    "actual": actual,
                }
            )
    return {
        **classification_metrics(expected, predicted),
        "evaluation_scope": "response parsing and strict verdict schema only; not semantic grounding",
        "errors": errors,
    }


def evaluate_semantic_cases(
    cases: list[SemanticCase],
    predictions: dict[str, str] | None = None,
) -> SemanticEvaluation:
    """Score external verdicts against manual source/answer labels if supplied."""
    if predictions is None:
        return {
            "status": "not_scored",
            "case_count": len(cases),
            "precision_recall": None,
            "reason": (
                "No observed verifier predictions supplied. The offline protocol "
                "fixtures do not measure semantic grounding."
            ),
        }
    expected_ids = {case["case_id"] for case in cases}
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        extra = sorted(set(predictions) - expected_ids)
        raise ValueError(
            f"prediction IDs must exactly match semantic cases "
            f"(missing={missing}, extra={extra})"
        )
    if any(label not in VERDICT_LABELS for label in predictions.values()):
        raise ValueError(f"predictions must use labels: {', '.join(VERDICT_LABELS)}")
    expected = [case["expected_verdict"] for case in cases]
    predicted = [predictions[case["case_id"]] for case in cases]
    mismatches = [
        {
            "case_id": case["case_id"],
            "expected": case["expected_verdict"],
            "predicted": predictions[case["case_id"]],
        }
        for case in cases
        if predictions[case["case_id"]] != case["expected_verdict"]
    ]
    return {
        "status": "scored",
        "evaluation_scope": "semantic labels vs supplied observed verifier verdicts",
        "precision_recall": classification_metrics(expected, predicted),
        "errors": mismatches,
    }


def collect_live_predictions(
    cases: list[SemanticCase],
    client: LLMClient,
) -> tuple[dict[str, str], list[LivePredictionError]]:
    """Record one actual verifier verdict per case without exposing error text."""
    verifier = AnswerGroundingVerifier()
    predictions: dict[str, str] = {}
    failures: list[LivePredictionError] = []
    for case in cases:
        try:
            predictions[case["case_id"]] = verifier.verify(
                case["query"],
                case["answer"],
                case["source_chunks"],
                client.generate,
            )
        except (GroundingVerificationError, LLMError) as error:
            predictions[case["case_id"]] = "error"
            failures.append(
                {"case_id": case["case_id"], "error_type": type(error).__name__}
            )
    return predictions, failures


def load_predictions(path: Path) -> dict[str, str]:
    """Load a JSON object mapping semantic case IDs to observed verdicts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("case_predictions"), dict
    ):
        raise ValueError("prediction file must contain a case_predictions object")
    return payload["case_predictions"]


def main() -> int:
    """Run protocol checks and optionally score or collect semantic observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    prediction_group = parser.add_mutually_exclusive_group()
    prediction_group.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Score a JSON file containing observed case_predictions.",
    )
    prediction_group.add_argument(
        "--live",
        action="store_true",
        help="Call the configured verifier once per labeled case and record results.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    dataset_path = (args.dataset or PROJECT_ROOT / DEFAULT_DATASET_PATH).resolve()
    output_path = (args.output or PROJECT_ROOT / DEFAULT_OUTPUT_PATH).resolve()
    client: LLMClient | None = None
    try:
        dataset = validate_dataset(json.loads(dataset_path.read_text(encoding="utf-8")))
        live_failures: list[LivePredictionError] = []
        live_metadata: dict[str, str | int] | None = None
        if args.predictions:
            predictions = load_predictions(args.predictions)
            prediction_source = "supplied_observations"
        elif args.live:
            client = LLMClient()
            predictions, live_failures = collect_live_predictions(
                dataset["cases"], client
            )
            prediction_source = "live_llm_verifier"
            live_metadata = {
                "provider": "DeepSeek",
                "model": client.model,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "request_count": len(dataset["cases"]),
            }
        else:
            predictions = None
            prediction_source = "none"
        report: dict[str, Any] = {
            "dataset_id": dataset["dataset_id"],
            "dataset_size": {
                "semantic_cases": len(dataset["cases"]),
                "protocol_cases": len(dataset["protocol_cases"]),
            },
            "protocol_evaluation": evaluate_protocol_cases(dataset["protocol_cases"]),
            "semantic_evaluation": evaluate_semantic_cases(
                dataset["cases"], predictions
            ),
            "semantic_prediction_source": prediction_source,
            "case_predictions": predictions,
            "live_evaluation": (
                {
                    **live_metadata,
                    "predictions": predictions,
                    "request_errors": live_failures,
                }
                if live_metadata is not None
                else None
            ),
            "limitations": [
                "Protocol precision/recall evaluate strict response parsing only.",
                "Semantic labels are a small manually authored sample and do not establish system-wide quality.",
                "Without observed predictions, semantic precision/recall are not measured.",
                "Mocked protocol fixtures do not invoke an LLM or test its reasoning.",
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except LLMError as error:
        print(
            f"Grounding live evaluation unavailable: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(
            f"Grounding evaluation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    finally:
        if client is not None:
            client.close()

    protocol = report["protocol_evaluation"]
    semantic = report["semantic_evaluation"]
    print(f"Dataset: {report['dataset_id']}")
    print(
        "Protocol-only: "
        f"accuracy={protocol['accuracy']:.3f}, "
        f"macro precision={protocol['macro_precision']:.3f}, "
        f"macro recall={protocol['macro_recall']:.3f}, "
        f"errors={len(protocol['errors'])}"
    )
    if semantic["status"] == "scored":
        metrics = semantic["precision_recall"]
        print(
            "Semantic evaluation: "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"macro precision={metrics['macro_precision']:.3f}, "
            f"macro recall={metrics['macro_recall']:.3f}"
        )
    else:
        print(
            f"Semantic evaluation: NOT SCORED ({semantic['case_count']} labeled cases; "
            "no observed predictions supplied)"
        )
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
