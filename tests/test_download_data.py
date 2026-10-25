"""Tests for reproducible AG News acquisition and raw-record validation."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

from scripts.download_data import (
    make_record,
    read_records,
    validate_corpus,
    write_corpus,
)


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indexes: Iterable[int]) -> list[dict[str, object]]:
        return [self.rows[index] for index in indexes]


def test_write_corpus_keeps_stable_ids_and_requested_source_rows(tmp_path) -> None:
    dataset = FakeDataset(
        [
            {"text": "First authentic dataset row", "label": 0},
            {"text": "Second authentic dataset row", "label": 1},
        ]
    )
    output = tmp_path / "raw.jsonl"

    assert write_corpus(dataset, output, document_count=2) == 2
    records = read_records(output)

    assert [record["document_id"] for record in records] == [
        "ag_news_train_000000",
        "ag_news_train_000001",
    ]
    assert [record["text"] for record in records] == [
        "First authentic dataset row",
        "Second authentic dataset row",
    ]
    validate_corpus(records, minimum_count=2)
    first_line = output.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line)["source"] == "fancyzhx/ag_news"


def test_write_corpus_rejects_unavailable_source_rows(tmp_path) -> None:
    with pytest.raises(ValueError, match="Source split contains"):
        write_corpus(FakeDataset([{"text": "one"}]), tmp_path / "raw.jsonl", 2)


def test_raw_validation_rejects_duplicate_document_ids() -> None:
    record = make_record({"text": "same text", "label": 0}, 1)
    with pytest.raises(RuntimeError, match="duplicate document IDs"):
        validate_corpus([record, record], minimum_count=2)
