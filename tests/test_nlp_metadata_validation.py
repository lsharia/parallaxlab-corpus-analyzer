"""Tests for metadata filter validation evidence helpers."""

from __future__ import annotations

import pytest

from scripts.validate_nlp_metadata import (
    build_topic_review_template,
    validate_retrieval_results,
)


def test_topic_review_template_leaves_human_judgments_pending() -> None:
    topic_evidence = [
        {
            "topic_id": 3,
            "top_words": "market, shares, company",
            "assessment": "coherent",
            "observation": "Automated observation.",
            "representative_documents": [
                {
                    "document_id": "doc-1",
                    "topic_probability": 0.8,
                    "text_preview": "Company shares rose.",
                }
            ],
        }
    ]

    template = build_topic_review_template(topic_evidence)

    assert template["review_status"] == "pending_human_review"
    assert template["topics"][0]["human_coherence_assessment"] is None
    assert template["topics"][0]["human_rationale"] == ""
    assert "Automated observation" not in str(template)
    assert template["topics"][0]["representative_documents"] == topic_evidence[0][
        "representative_documents"
    ]


def test_topic_review_template_requires_topic_evidence() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_topic_review_template([])


def test_retrieval_validation_accepts_matching_metadata() -> None:
    results = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source": "news",
            "text": "An example.",
            "distance": 0.2,
        }
    ]
    metadata = {
        "chunk-1": {
            "document_id": "doc-1",
            "topic_id": 8,
            "sentiment_label": "positive",
        }
    }

    check = validate_retrieval_results(results, metadata, 8, "positive", 5)

    assert check["result_count"] == 1
    assert check["all_results_match"] is True


@pytest.mark.parametrize(
    ("topic_id", "sentiment_label", "metadata"),
    [
        (8, None, {"document_id": "doc-1", "topic_id": 2}),
        (None, "negative", {"document_id": "doc-1", "sentiment_label": "positive"}),
    ],
)
def test_retrieval_validation_rejects_filter_mismatches(
    topic_id, sentiment_label, metadata
) -> None:
    result = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "source": "news",
        "text": "An example.",
        "distance": 0.2,
    }

    with pytest.raises(RuntimeError, match="violates"):
        validate_retrieval_results(
            [result], {"chunk-1": metadata}, topic_id, sentiment_label, 5
        )


def test_retrieval_validation_rejects_empty_result_set() -> None:
    with pytest.raises(RuntimeError, match="returned no results"):
        validate_retrieval_results([], {}, None, None, 5)