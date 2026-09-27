"""Semantic retrieval over the persistent ChromaDB chunk collection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings import DEFAULT_MODEL_NAME, load_embedding_model
from src.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DATABASE_PATH


class SemanticRetriever:
    """Reuse a sentence-transformer model and persistent Chroma collection."""

    def __init__(
        self,
        database_path: Path = Path(DEFAULT_DATABASE_PATH),
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        model: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        self.database_path = Path(database_path).resolve()
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self.last_latency_seconds = 0.0

        if not self.database_path.is_dir():
            raise FileNotFoundError(
                f"ChromaDB database directory does not exist: {self.database_path}"
            )

        try:
            import chromadb
            from chromadb.errors import NotFoundError
        except ImportError as error:
            raise RuntimeError("The 'chromadb' package is required.") from error

        self.client = chromadb.PersistentClient(path=str(self.database_path))
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except NotFoundError as error:
            raise LookupError(
                f"ChromaDB collection {collection_name!r} does not exist in "
                f"{self.database_path}."
            ) from error

        self.model = model or load_embedding_model(model_name, device=device)

    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Embed a query and return its nearest stored chunks with cosine distances."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        collection_size = self.collection.count()
        if collection_size == 0:
            self.last_latency_seconds = 0.0
            return []

        result_count = min(top_k, collection_size)
        started_at = time.perf_counter()
        try:
            query_vectors = np.asarray(
                self.model.encode(
                    [query_text.strip()],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        except Exception as error:
            raise RuntimeError(f"Query embedding failed: {error}") from error

        if query_vectors.ndim != 2 or query_vectors.shape[0] != 1:
            raise ValueError(
                f"Query model must return one vector; got shape {query_vectors.shape}."
            )
        vector = query_vectors[0]
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("Query embedding must be a non-empty finite vector")

        try:
            matches = self.collection.query(
                query_embeddings=[vector.tolist()],
                n_results=result_count,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            raise RuntimeError(f"ChromaDB semantic query failed: {error}") from error
        self.last_latency_seconds = time.perf_counter() - started_at

        ids = matches["ids"][0]
        documents = matches["documents"][0]
        metadatas = matches["metadatas"][0]
        distances = matches["distances"][0]
        return [
            {
                "chunk_id": chunk_id,
                "document_id": (metadata or {}).get("document_id"),
                "source": (metadata or {}).get("source"),
                "text": document,
                "distance": float(distance),
            }
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]


def retrieve(
    query_text: str,
    top_k: int = 5,
    *,
    database_path: Path = Path(DEFAULT_DATABASE_PATH),
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """Convenience function for a single semantic search operation."""
    retriever = SemanticRetriever(
        database_path=database_path,
        collection_name=collection_name,
        model_name=model_name,
        device=device,
        model=model,
    )
    return retriever.retrieve(query_text, top_k=top_k)


def main() -> int:
    """Run one semantic query against the local ChromaDB collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    database_path = (args.database or project_root / DEFAULT_DATABASE_PATH).resolve()
    try:
        retriever = SemanticRetriever(
            database_path=database_path,
            collection_name=args.collection,
            model_name=args.model,
            device=args.device,
        )
        results = retriever.retrieve(args.query, top_k=args.top_k)
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as error:
        print(f"Retrieval failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print(f"Latency: {retriever.last_latency_seconds:.4f} seconds")
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result["text"].split())[:240]
        print(
            f"{rank}. chunk_id={result['chunk_id']} "
            f"distance={result['distance']:.6f}\n   {preview}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
