"""Deterministically split the validated clean corpus into text chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_PATH = "data/processed/clean_corpus.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/chunks.parquet"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
SEPARATORS = ("\n\n", "\n", " ", "")


def _split_recursively(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Split text using the largest available natural boundary first."""
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    if separator == "":
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    pieces = text.split(separator)
    if len(pieces) == 1:
        return _split_recursively(text, chunk_size, separators[1:])

    fragments: list[str] = []
    for index, piece in enumerate(pieces):
        if not piece:
            continue
        fragment = piece if index == len(pieces) - 1 else piece + separator
        if len(fragment) <= chunk_size:
            fragments.append(fragment)
        else:
            fragments.extend(_split_recursively(fragment, chunk_size, separators[1:]))
    return fragments


def _merge_with_overlap(
    pieces: Iterable[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Merge split pieces into bounded chunks with deterministic character overlap."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
            continue

        candidate = current + piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        overlap = current[-chunk_overlap:] if chunk_overlap else ""
        current = overlap + piece
        if len(current) > chunk_size:
            current = current[:chunk_size]

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def split_text(
    text: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split one document with recursive boundaries and character overlap.

    The defaults use 500 characters per chunk and 50 characters of overlap.
    Empty or non-string input returns no chunks. Chunks are deterministic,
    non-empty, and never exceed ``chunk_size`` characters.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    pieces = _split_recursively(text, chunk_size, SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Chunk one document while preserving its document ID and source metadata."""
    document_id = document.get("document_id")
    source = document.get("source")
    chunks = split_text(document.get("text"), chunk_size, chunk_overlap)
    return [
        {
            "chunk_id": f"{document_id}_chunk_{index:04d}",
            "document_id": document_id,
            "source": source,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks)
    ]


def process_dataset(
    dataframe: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[Any, dict[str, float | int]]:
    """Chunk every row in a clean DataFrame and return output plus statistics."""
    chunks: list[dict[str, Any]] = []
    source_documents = len(dataframe)
    for record in dataframe.to_dict("records"):
        chunks.extend(chunk_document(record, chunk_size, chunk_overlap))

    import pandas as pd

    chunk_dataframe = pd.DataFrame(
        chunks, columns=["chunk_id", "document_id", "source", "text"]
    )
    chunk_lengths = chunk_dataframe["text"].str.len() if chunks else pd.Series(dtype="int64")
    statistics = {
        "source_documents": source_documents,
        "chunks": len(chunks),
        "average_chunks_per_document": len(chunks) / source_documents if source_documents else 0.0,
        "average_chunk_length": float(chunk_lengths.mean()) if chunks else 0.0,
        "minimum_chunk_length": int(chunk_lengths.min()) if chunks else 0,
        "maximum_chunk_length": int(chunk_lengths.max()) if chunks else 0,
    }
    return chunk_dataframe, statistics


def load_and_chunk_dataset(
    input_path: Path,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, float | int]:
    """Load clean Parquet, chunk it, save chunks Parquet, and return statistics."""
    import pandas as pd

    dataframe = pd.read_parquet(input_path)
    required_columns = {"document_id", "source", "text"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    chunk_dataframe, statistics = process_dataset(dataframe, chunk_size, chunk_overlap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dataframe.to_parquet(output_path, index=False)
    return statistics


def main() -> int:
    """Run chunking from the command line and report useful statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = (args.input or project_root / DEFAULT_INPUT_PATH).resolve()
    output_path = (args.output or project_root / DEFAULT_OUTPUT_PATH).resolve()

    try:
        statistics = load_and_chunk_dataset(
            input_path, output_path, args.chunk_size, args.chunk_overlap
        )
    except (OSError, ValueError, ImportError) as error:
        print(f"Chunking failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print("Chunking statistics:")
    print(f"Source documents: {statistics['source_documents']:,}")
    print(f"Chunks: {statistics['chunks']:,}")
    print(f"Average chunks per document: {statistics['average_chunks_per_document']:.2f}")
    print(f"Average chunk length: {statistics['average_chunk_length']:.1f} characters")
    print(f"Minimum chunk length: {statistics['minimum_chunk_length']} characters")
    print(f"Maximum chunk length: {statistics['maximum_chunk_length']} characters")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
