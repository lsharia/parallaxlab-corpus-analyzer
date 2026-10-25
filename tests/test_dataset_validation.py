"""Tests for the required minimum clean-corpus size."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_dataset import validate_dataset


def test_clean_corpus_must_contain_at_least_5000_documents(tmp_path: Path) -> None:
    dataset_path = tmp_path / "clean_corpus.parquet"
    pd.DataFrame(
        {
            "document_id": [f"doc-{index}" for index in range(4_999)],
            "source": ["test"] * 4_999,
            "text": ["valid text"] * 4_999,
        }
    ).to_parquet(dataset_path, index=False)

    with pytest.raises(ValueError, match="at least 5,000 are required"):
        validate_dataset(dataset_path)
