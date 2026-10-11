"""Tests for deterministic topic modeling over cleaned documents."""

from __future__ import annotations

import pandas as pd
import pytest

from src.topic_modeling import (
    UNASSIGNED_TOPIC_ID,
    TopicModelConfig,
    fit_topic_model,
    run_topic_modeling,
    validate_topic_input,
    validate_topic_outputs,
)


@pytest.fixture
def news_documents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document_id": [f"doc-{index}" for index in range(6)],
            "text": [
                "Stocks shares market investors trading company earnings",
                "Markets rallied as stocks climbed after company earnings",
                "Investors bought shares while the market gained value",
                "Football team scored goals during the championship match",
                "The coach praised players after a dramatic football game",
                "Championship players won the match with a late goal",
            ],
        }
    )


def test_topic_model_is_deterministic(news_documents: pd.DataFrame) -> None:
    config = TopicModelConfig(n_topics=2, random_state=19, max_iter=5)

    first = fit_topic_model(news_documents, config)
    second = fit_topic_model(news_documents, config)

    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_assignments_have_valid_topics_and_probabilities(
    news_documents: pd.DataFrame,
) -> None:
    result = fit_topic_model(news_documents, TopicModelConfig(n_topics=2, max_iter=5))

    validate_topic_outputs(result.assignments, result.summary)
    assert result.effective_n_topics == 2
    assert result.assignments["topic_id"].between(0, 1).all()
    assert result.assignments["topic_probability"].between(0.0, 1.0).all()
    assert result.assignments["topic_probability"].gt(0.0).all()


def test_topic_summary_contains_representative_words(
    news_documents: pd.DataFrame,
) -> None:
    result = fit_topic_model(news_documents, TopicModelConfig(n_topics=2, max_iter=5))

    assert list(result.summary.columns) == ["topic_id", "top_words"]
    assert len(result.summary) == 2
    assert result.summary["top_words"].str.len().gt(0).all()
    assert all(len(words.split(", ")) <= 10 for words in result.summary["top_words"])


def test_empty_short_punctuation_and_numeric_documents_are_unassigned() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["empty", "short", "punctuation", "numbers", "usable"],
            "text": ["", "a I", "!!! ??? ...", "123 456 2024", "Markets saw strong company earnings"],
        }
    )

    result = fit_topic_model(documents, TopicModelConfig(n_topics=3, max_iter=3))

    assert result.assignments.loc[:3, "topic_id"].eq(UNASSIGNED_TOPIC_ID).all()
    assert result.assignments.loc[:3, "topic_probability"].eq(0.0).all()
    assert result.assignments.loc[4, "topic_id"] == 0
    assert result.effective_n_topics == 1


def test_corpus_without_usable_terms_returns_empty_summary() -> None:
    documents = pd.DataFrame(
        {"document_id": ["one", "two"], "text": ["", "123 !!! the and"]}
    )

    result = fit_topic_model(documents)

    assert result.assignments["topic_id"].eq(UNASSIGNED_TOPIC_ID).all()
    assert result.summary.empty
    assert result.model is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_topics": 0},
        {"n_topics": True},
        {"random_state": -1},
        {"random_state": 1.5},
        {"max_iter": 0},
        {"max_features": 0},
        {"top_words": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TopicModelConfig(**kwargs)


def test_small_corpus_reduces_topic_count_to_usable_documents() -> None:
    documents = pd.DataFrame(
        {
            "document_id": ["one", "two"],
            "text": ["technology software computer hardware", "election government policy voting"],
        }
    )

    result = fit_topic_model(documents, TopicModelConfig(n_topics=10, max_iter=3))

    assert result.effective_n_topics == 2
    assert len(result.summary) == 2
    assert result.assignments["topic_id"].between(0, 1).all()


def test_input_schema_and_identifiers_are_validated() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_topic_input(pd.DataFrame({"document_id": ["one"]}))
    with pytest.raises(ValueError, match="duplicate"):
        validate_topic_input(
            pd.DataFrame({"document_id": ["same", "same"], "text": ["a", "b"]})
        )
    with pytest.raises(ValueError, match="missing document IDs"):
        validate_topic_input(
            pd.DataFrame({"document_id": [None], "text": ["some text"]})
        )
    with pytest.raises(ValueError, match="must be strings or null"):
        validate_topic_input(pd.DataFrame({"document_id": ["one"], "text": [123]}))


def test_output_schema_and_probability_validation() -> None:
    assignments = pd.DataFrame(
        {"document_id": ["one"], "topic_id": [0], "topic_probability": [1.01]}
    )
    summary = pd.DataFrame({"topic_id": [0], "top_words": ["news, market"]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_topic_outputs(assignments, summary)

    with pytest.raises(ValueError, match="missing columns"):
        validate_topic_outputs(pd.DataFrame({"document_id": ["one"]}), summary)


def test_run_writes_schema_valid_artifacts_and_reuses_matching_results(
    news_documents: pd.DataFrame,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "clean_corpus.parquet"
    assignments_path = tmp_path / "topic_assignments.parquet"
    summary_path = tmp_path / "topic_summary.parquet"
    news_documents.to_parquet(input_path, index=False)
    config = TopicModelConfig(n_topics=2, max_iter=3)

    generated = run_topic_modeling(input_path, assignments_path, summary_path, config)
    assert generated["cached"] is False
    assert generated["manifest_path"].is_file()

    def fail_if_retrained(*args, **kwargs):
        raise AssertionError("Matching topic artifacts should be reused")

    monkeypatch.setattr("src.topic_modeling.fit_topic_model", fail_if_retrained)
    reused = run_topic_modeling(input_path, assignments_path, summary_path, config)

    assert reused["cached"] is True
    assignments = pd.read_parquet(assignments_path)
    summary = pd.read_parquet(summary_path)
    assert list(assignments.columns) == ["document_id", "topic_id", "topic_probability"]
    assert list(summary.columns) == ["topic_id", "top_words"]
    validate_topic_outputs(assignments, summary)


def test_cached_artifacts_are_invalidated_when_configuration_changes(
    news_documents: pd.DataFrame,
    tmp_path,
) -> None:
    input_path = tmp_path / "clean_corpus.parquet"
    assignments_path = tmp_path / "topic_assignments.parquet"
    summary_path = tmp_path / "topic_summary.parquet"
    news_documents.to_parquet(input_path, index=False)

    first = run_topic_modeling(
        input_path, assignments_path, summary_path, TopicModelConfig(n_topics=2, max_iter=2)
    )
    changed = run_topic_modeling(
        input_path, assignments_path, summary_path, TopicModelConfig(n_topics=3, max_iter=2)
    )

    assert first["cached"] is False
    assert changed["cached"] is False
    assert changed["topic_count"] == 3