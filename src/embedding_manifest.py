"""Fingerprint source text so embedding manifests cannot be reused stale."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import json


def chunk_source_fingerprint(records: Iterable[tuple[object, object]]) -> str:
    """Return a stable SHA-256 digest for chunk IDs and their exact source text."""
    normalized: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        try:
            chunk_id, text = record
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Chunk fingerprint record {index} must contain two values."
            ) from error
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Chunk fingerprint record {index} has an invalid chunk ID.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk fingerprint record {index} has invalid source text.")
        normalized.append((chunk_id, text))

    if len({chunk_id for chunk_id, _ in normalized}) != len(normalized):
        raise ValueError("Chunk fingerprint records contain duplicate chunk IDs.")

    payload = json.dumps(
        sorted(normalized),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()
