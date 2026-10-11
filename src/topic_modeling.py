"""Fit reproducible LDA topics over the cleaned document corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


DEFAULT_INPUT_PATH = "data/processed/clean_corpus.parquet"
DEFAULT_ASSIGNMENTS_PATH = "data/processed/topic_assignments.parquet"
DEFAULT_SUMMARY_PATH = "data/processed/topic_summary.parquet"
DEFAULT_N_TOPICS = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 10
DEFAULT_MAX_FEATURES = 10_000
DEFAULT_TOP_WORDS = 10
UNASSIGNED_TOPIC_ID = -1
MANIFEST_VERSION = 1
VECTORIZER_VERSION = 1
NEWS_STOP_WORDS = ENGLISH_STOP_WORDS.union(
    {"ap", "com", "fullquote", "http", "https", "reuters", "said", "ticker", "www"}
)
REQUIRED_INPUT_COLUMNS = {"document_id", "text"}
REQUIRED_ASSIGNMENT_COLUMNS = {"document_id", "topic_id", "topic_probability"}
REQUIRED_SUMMARY_COLUMNS = {"topic_id", "top_words"}


@dataclass(frozen=True)
class TopicModelConfig:
    """Settings for vectorization and deterministic LDA fitting."""

    n_topics: int = DEFAULT_N_TOPICS
    random_state: int = DEFAULT_RANDOM_STATE
    max_iter: int = DEFAULT_MAX_ITER
    max_features: int = DEFAULT_MAX_FEATURES
    top_words: int = DEFAULT_TOP_WORDS

    def __post_init__(self) -> None:
        for name in ("n_topics", "max_iter", "max_features", "top_words"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or not 0 <= self.random_state <= 2**32 - 1
        ):
            raise ValueError("random_state must be an integer between 0 and 2**32 - 1")


@dataclass
class TopicModelResult:
    """Document assignments, topic labels, and fitted model components."""

    assignments: pd.DataFrame
    summary: pd.DataFrame
    model: LatentDirichletAllocation | None
    vectorizer: CountVectorizer | None
    effective_n_topics: int


def validate_topic_input(documents: pd.DataFrame) -> None:
    """Validate the required clean-corpus columns and document identifiers."""
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


def validate_topic_outputs(assignments: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Validate the public assignment and summary table schemas and values."""
    if not isinstance(assignments, pd.DataFrame) or not isinstance(summary, pd.DataFrame):
        raise ValueError("Topic outputs must be pandas DataFrames")
    missing_assignments = REQUIRED_ASSIGNMENT_COLUMNS - set(assignments.columns)
    if missing_assignments:
        raise ValueError(
            f"Assignments are missing columns: {sorted(missing_assignments)}"
        )
    missing_summary = REQUIRED_SUMMARY_COLUMNS - set(summary.columns)
    if missing_summary:
        raise ValueError(f"Summary is missing columns: {sorted(missing_summary)}")
    if assignments["document_id"].isna().any() or not assignments["document_id"].is_unique:
        raise ValueError("Assignment document IDs must be present and unique")
    if summary["topic_id"].isna().any() or not summary["topic_id"].is_unique:
        raise ValueError("Summary topic IDs must be present and unique")
    if summary["topic_id"].lt(0).any():
        raise ValueError("Summary topic IDs must be non-negative")
    if summary["top_words"].map(lambda value: not isinstance(value, str) or not value.strip()).any():
        raise ValueError("Every topic summary must contain representative words")

    try:
        probabilities = assignments["topic_probability"].astype(float)
        topic_ids = assignments["topic_id"].astype(int)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Topic IDs and probabilities must be numeric") from error
    if not probabilities.map(math.isfinite).all() or not probabilities.between(0.0, 1.0).all():
        raise ValueError("Topic probabilities must be finite values between 0 and 1")
    if not assignments["topic_id"].map(
        lambda value: isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    ).all():
        raise ValueError("Topic IDs must be integers")

    assigned = topic_ids.ge(0)
    if (topic_ids.loc[~assigned] != UNASSIGNED_TOPIC_ID).any():
        raise ValueError(f"Unassigned topic IDs must equal {UNASSIGNED_TOPIC_ID}")
    if probabilities.loc[~assigned].ne(0.0).any():
        raise ValueError("Unassigned documents must have probability 0")
    if assigned.any() and not set(topic_ids.loc[assigned]).issubset(
        set(summary["topic_id"].astype(int))
    ):
        raise ValueError("Assignments reference topics missing from the summary")


def fit_topic_model(
    documents: pd.DataFrame,
    config: TopicModelConfig | None = None,
) -> TopicModelResult:
    """Vectorize cleaned text, fit LDA, and assign every input document."""
    validate_topic_input(documents)
    config = config or TopicModelConfig()
    document_ids = documents["document_id"].reset_index(drop=True)
    texts = documents["text"].map(lambda value: value if isinstance(value, str) else "").tolist()

    vectorizer = CountVectorizer(
        stop_words=sorted(NEWS_STOP_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
        max_features=config.max_features,
    )
    analyze = vectorizer.build_analyzer()
    eligible_indexes = [index for index, text in enumerate(texts) if analyze(text)]
    assignment_topic_ids = [UNASSIGNED_TOPIC_ID] * len(texts)
    assignment_probabilities = [0.0] * len(texts)

    if not eligible_indexes:
        assignments = pd.DataFrame(
            {
                "document_id": document_ids,
                "topic_id": assignment_topic_ids,
                "topic_probability": assignment_probabilities,
            }
        )
        summary = pd.DataFrame(
            {
                "topic_id": pd.Series(dtype="int64"),
                "top_words": pd.Series(dtype="str"),
            }
        )
        validate_topic_outputs(assignments, summary)
        return TopicModelResult(assignments, summary, None, None, 0)

    eligible_texts = [texts[index] for index in eligible_indexes]
    term_counts = vectorizer.fit_transform(eligible_texts)
    effective_n_topics = min(config.n_topics, len(eligible_indexes))
    model = LatentDirichletAllocation(
        n_components=effective_n_topics,
        max_iter=config.max_iter,
        learning_method="batch",
        random_state=config.random_state,
    )
    document_topic_distributions = model.fit_transform(term_counts)
    for index, distribution in zip(eligible_indexes, document_topic_distributions):
        topic_id = int(distribution.argmax())
        assignment_topic_ids[index] = topic_id
        assignment_probabilities[index] = float(distribution[topic_id])

    feature_names = vectorizer.get_feature_names_out()
    topic_rows = []
    for topic_id, component in enumerate(model.components_):
        top_indexes = component.argsort()[::-1][: min(config.top_words, len(feature_names))]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "top_words": ", ".join(feature_names[top_indexes].tolist()),
            }
        )

    assignments = pd.DataFrame(
        {
            "document_id": document_ids,
            "topic_id": assignment_topic_ids,
            "topic_probability": assignment_probabilities,
        }
    )
    summary = pd.DataFrame(topic_rows, columns=["topic_id", "top_words"])
    validate_topic_outputs(assignments, summary)
    return TopicModelResult(
        assignments, summary, model, vectorizer, effective_n_topics
    )


def _corpus_fingerprint(documents: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for document_id, text in zip(documents["document_id"], documents["text"]):
        encoded = json.dumps(
            [document_id, text if isinstance(text, str) else ""],
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def run_topic_modeling(
    input_path: Path,
    assignments_path: Path,
    summary_path: Path,
    config: TopicModelConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create or reuse topic output Parquet files for a corpus/config pair."""
    config = config or TopicModelConfig()
    documents = pd.read_parquet(input_path)
    validate_topic_input(documents)
    fingerprint = _corpus_fingerprint(documents)
    manifest_path = assignments_path.with_name("topic_model_manifest.json")
    expected_manifest = {
        "version": MANIFEST_VERSION,
        "vectorizer_version": VECTORIZER_VERSION,
        "corpus_sha256": fingerprint,
        "config": asdict(config),
        "document_count": len(documents),
    }

    if not force and manifest_path.is_file() and assignments_path.is_file() and summary_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assignments = pd.read_parquet(assignments_path)
            summary = pd.read_parquet(summary_path)
            validate_topic_outputs(assignments, summary)
            ids_match = assignments["document_id"].tolist() == documents["document_id"].tolist()
            if manifest == expected_manifest and ids_match:
                return {
                    "assignments_path": assignments_path,
                    "summary_path": summary_path,
                    "manifest_path": manifest_path,
                    "document_count": len(assignments),
                    "topic_count": len(summary),
                    "cached": True,
                }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    result = fit_topic_model(documents, config)
    validate_topic_outputs(result.assignments, result.summary)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.assignments.to_parquet(assignments_path, index=False)
    result.summary.to_parquet(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "assignments_path": assignments_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "document_count": len(result.assignments),
        "topic_count": result.effective_n_topics,
        "cached": False,
    }


def main() -> int:
    """Run LDA on the cleaned corpus and write ignored processed artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--assignments-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--n-topics", type=int, default=DEFAULT_N_TOPICS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--top-words", type=int, default=DEFAULT_TOP_WORDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = (args.input or project_root / DEFAULT_INPUT_PATH).resolve()
    assignments_path = (
        args.assignments_output or project_root / DEFAULT_ASSIGNMENTS_PATH
    ).resolve()
    summary_path = (args.summary_output or project_root / DEFAULT_SUMMARY_PATH).resolve()
    try:
        config = TopicModelConfig(
            n_topics=args.n_topics,
            random_state=args.random_state,
            max_iter=args.max_iter,
            max_features=args.max_features,
            top_words=args.top_words,
        )
        report = run_topic_modeling(
            input_path,
            assignments_path,
            summary_path,
            config=config,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"Topic modeling failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    action = "Reused" if report["cached"] else "Generated"
    print(f"{action} topics for {report['document_count']:,} documents")
    print(f"Topics: {report['topic_count']}")
    print(f"Assignments: {report['assignments_path']}")
    print(f"Summary: {report['summary_path']}")
    if not report["cached"]:
        print(f"Cache manifest: {report['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())