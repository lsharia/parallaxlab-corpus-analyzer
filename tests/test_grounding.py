"""Tests for answer-to-source grounding verification."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from src.grounding import (
    AnswerGroundingVerifier,
    GroundingVerificationError,
    build_grounding_messages,
)


SOURCE_CHUNKS = [
    {
        "chunk_id": "chunk-1",
        "source": "news",
        "text": "The company reported revenue of $12 million in 2025.",
    }
]


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("supported", "supported"),
        ("unsupported", "unsupported"),
        ("insufficient_evidence", "insufficient_evidence"),
    ],
)
def test_verifier_returns_structured_source_verdict(
    verdict: str,
    expected: str,
) -> None:
    captured: list[Sequence[Mapping[str, Any]]] = []

    def generate(messages: Sequence[Mapping[str, Any]]) -> str:
        captured.append(messages)
        return json.dumps({"verdict": verdict})

    result = AnswerGroundingVerifier().verify(
        "What was reported?",
        "Revenue was $12 million.",
        SOURCE_CHUNKS,
        generate,
    )

    assert result == expected
    assert len(captured) == 1
    assert "every factual claim" in captured[0][0]["content"]
    assert "Revenue was $12 million." in captured[0][1]["content"]
    assert "The company reported revenue" in captured[0][1]["content"]


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        '{"verdict":"uncertain"}',
        '{"verdict":"supported","extra":"ignored"}',
        '{"verdict":[]}',
        "[]",
        "null",
    ],
)
def test_verifier_rejects_malformed_or_ambiguous_results(response: str) -> None:
    with pytest.raises(GroundingVerificationError):
        AnswerGroundingVerifier().verify(
            "Question",
            "Candidate answer.",
            SOURCE_CHUNKS,
            lambda _messages: response,
        )


def test_grounding_prompt_requires_source_chunks() -> None:
    with pytest.raises(ValueError, match="source chunk"):
        build_grounding_messages("Question", "Answer", [{"text": "  "}])
