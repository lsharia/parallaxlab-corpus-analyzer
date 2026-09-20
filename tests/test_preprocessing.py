"""Unit tests for the modular preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    clean_html,
    clean_text,
    clean_whitespace,
    is_valid_language,
    normalize_unicode,
    process_records,
    save_clean_dataset,
)


def test_clean_html_removes_tags_and_preserves_text() -> None:
    assert clean_html("<p>Hello <strong>world</strong>.</p>") == "Hello world."


def test_clean_html_decodes_character_references() -> None:
    assert clean_html("<p>AT&amp;T&nbsp;news</p>") == "AT&T\u00a0news"


def test_normalize_unicode_uses_nfc() -> None:
    decomposed = "e\u0301"
    assert normalize_unicode(decomposed) == "é"
    assert normalize_unicode(decomposed) == normalize_unicode("é")


def test_clean_whitespace_collapses_spaces_tabs_and_newlines() -> None:
    messy = "  Keep   meaningful\ttext\nacross   lines.  "
    assert clean_whitespace(messy) == "Keep meaningful text across lines."


@pytest.mark.parametrize("value", [None, "", "   ", 123, {"text": "value"}])
def test_text_cleaning_handles_empty_and_non_string_values(value: object) -> None:
    assert clean_text(value) == ""


def test_is_valid_language_accepts_english_text() -> None:
    assert is_valid_language(
        "This is a detailed English news article about markets and technology."
    )


def test_is_valid_language_rejects_unsupported_language() -> None:
    assert not is_valid_language("这是一个关于市场和技术的中文新闻文章。")


@pytest.mark.parametrize("value", [None, "", "   ", 42])
def test_is_valid_language_rejects_empty_or_non_string_values(value: object) -> None:
    assert not is_valid_language(value)


def test_clean_text_combines_html_unicode_and_whitespace_cleanup() -> None:
    messy = "  <p>Cafe\u0301&nbsp;&nbsp;  <em>news</em>\narticle.</p>  "
    assert clean_text(messy) == "Café news article."


def test_process_records_filters_empty_and_unsupported_documents() -> None:
    records = [
        {"document_id": "keep", "source": "test", "text": "<p>English news about science.</p>"},
        {"document_id": "empty", "source": "test", "text": None},
        {"document_id": "foreign", "source": "test", "text": "这是中文新闻文章。"},
    ]

    cleaned, statistics = process_records(records)

    assert [record["document_id"] for record in cleaned] == ["keep"]
    assert cleaned[0]["source"] == "test"
    assert cleaned[0]["text"] == "English news about science."
    assert statistics["raw_documents"] == 3
    assert statistics["cleaned_documents"] == 1
    assert statistics["empty_documents_removed"] == 1
    assert statistics["language_filtered_documents"] == 1
    assert statistics["removed_documents"] == 2


def test_save_clean_dataset_writes_expected_schema(tmp_path) -> None:
    records = [
        {
            "document_id": "doc-1",
            "source": "test",
            "text": "A clean English document.",
        }
    ]
    output_path = tmp_path / "clean_corpus.parquet"

    save_clean_dataset(records, output_path)
    loaded = pd.read_parquet(output_path)

    assert list(loaded.columns) == ["document_id", "source", "text"]
    assert loaded.to_dict("records") == [records[0]]
