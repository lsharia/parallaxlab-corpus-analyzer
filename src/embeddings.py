"""Generate and persist sentence-transformer embeddings for text chunks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INPUT_PATH = "data/processed/chunks.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/embeddings.parquet"
DEFAULT_BATCH_SIZE = 32


def select_device(device: str = "auto") -> str:
    """Choose CUDA when available for auto mode, otherwise use CPU."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    try:
        import torch
    except ImportError:
        if device == "cuda":
            raise RuntimeError("CUDA was requested, but PyTorch is not installed.")
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    return device


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME, device: str = "auto") -> Any:
    """Load a sentence-transformers model on the selected compute device."""
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    selected_device = select_device(device)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "The sentence-transformers package is required to generate embeddings."
        ) from error

    return SentenceTransformer(model_name, device=selected_device)


def embed_texts(
    texts: Sequence[str],
    model: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress_bar: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Encode texts in batches and return vectors with timing statistics.

    Empty input is valid and returns an empty matrix. Individual entries must be
    non-empty strings so each input maps to exactly one output embedding.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not texts:
        dimension = model.get_sentence_embedding_dimension()
        return np.empty((0, dimension), dtype=np.float32), {
            "number_of_chunks": 0,
            "embedding_dimension": dimension,
            "total_embedding_time": 0.0,
            "average_time_per_chunk": 0.0,
            "chunks_per_second": 0.0,
        }

    invalid_indexes = [
        index
        for index, text in enumerate(texts)
        if not isinstance(text, str) or not text.strip()
    ]
    if invalid_indexes:
        raise ValueError(
            "Embedding input contains empty or non-string text at indexes: "
            f"{invalid_indexes[:10]}"
        )

    started_at = time.perf_counter()
    vectors = np.asarray(
        model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    elapsed = time.perf_counter() - started_at

    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise RuntimeError(
            f"Model returned shape {vectors.shape} for {len(texts)} input texts."
        )

    dimension = int(vectors.shape[1])
    number_of_chunks = len(texts)
    return vectors, {
        "number_of_chunks": number_of_chunks,
        "embedding_dimension": dimension,
        "total_embedding_time": elapsed,
        "average_time_per_chunk": elapsed / number_of_chunks,
        "chunks_per_second": number_of_chunks / elapsed if elapsed else 0.0,
    }


def generate_chunk_embeddings(
    chunks: Any,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "auto",
    show_progress_bar: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Generate one embedding per chunk and return a chunk_id mapping table."""
    required_columns = {"chunk_id", "text"}
    missing_columns = required_columns - set(chunks.columns)
    if missing_columns:
        raise ValueError(f"Chunk dataset is missing columns: {sorted(missing_columns)}")
    if chunks["chunk_id"].isna().any() or not chunks["chunk_id"].is_unique:
        raise ValueError("Chunk IDs must be present and unique.")

    texts = chunks["text"].tolist()
    model = load_embedding_model(model_name, device)
    selected_device = select_device(device)
    vectors, statistics = embed_texts(
        texts, model, batch_size=batch_size, show_progress_bar=show_progress_bar
    )
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Generated {len(vectors)} embeddings for {len(chunks)} chunks."
        )

    import pandas as pd

    embedding_table = pd.DataFrame(
        {
            "chunk_id": chunks["chunk_id"].tolist(),
            "embedding": [vector.tolist() for vector in vectors],
        }
    )
    metadata = {
        "model_name": model_name,
        "device": selected_device,
        "batch_size": batch_size,
        **statistics,
    }
    return embedding_table, metadata


def embed_chunk_dataset(
    input_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "auto",
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Read chunks Parquet, generate embeddings, save the mapping, and return metrics."""
    import pandas as pd

    chunks = pd.read_parquet(input_path)
    embedding_table, metadata = generate_chunk_embeddings(
        chunks,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        show_progress_bar=show_progress_bar,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_table.to_parquet(output_path, index=False)
    return metadata


def main() -> int:
    """Run chunk embedding generation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = (args.input or project_root / DEFAULT_INPUT_PATH).resolve()
    output_path = (args.output or project_root / DEFAULT_OUTPUT_PATH).resolve()

    try:
        metadata = embed_chunk_dataset(
            input_path,
            output_path,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"Embedding failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print("Embedding performance:")
    print(f"Model: {metadata['model_name']}")
    print(f"Number of chunks: {metadata['number_of_chunks']:,}")
    print(f"Batch size: {metadata['batch_size']}")
    print(f"Device: {metadata['device']}")
    print(f"Embedding dimension: {metadata['embedding_dimension']}")
    print(f"Total embedding time: {metadata['total_embedding_time']:.2f} seconds")
    print(f"Average time per chunk: {metadata['average_time_per_chunk']:.6f} seconds")
    print(f"Throughput: {metadata['chunks_per_second']:.2f} chunks/second")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
