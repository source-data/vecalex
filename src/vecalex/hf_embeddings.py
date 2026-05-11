"""HuggingFace-hosted entity embeddings.

This module provides a lightweight retrieval path for *precomputed* entity vectors
stored as parquet on HuggingFace.

The main performance trick is to avoid scanning many parquet files by using
Hive-style partitioning so DuckDB can prune which files it needs to touch.

Expected dataset layout (recommended):

  hf://datasets/<dataset>/entity_type=<A|I|S|...>/*.parquet

Optional extra partitioning for even fewer files per lookup:

  hf://datasets/<dataset>/entity_type=<...>/bucket=<0..bucket_count-1>/*.parquet

In both cases each parquet file should include at least:
  - entity_id (string)
  - vec       (list/array)
"""

from __future__ import annotations

from functools import lru_cache
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


def _extract_openalex_entity_type(entity_id: str) -> str | None:
    """Return the OpenAlex entity type letter from an ID.

    Examples
    --------
    - "https://openalex.org/A123" -> "A"
    - "https://openalex.org/S199078552" -> "S"
    """

    if not isinstance(entity_id, str):
        return None
    tail = entity_id.rsplit("/", 1)[-1]
    if not tail:
        return None
    t = tail[0]
    if t.isalpha():
        return t.upper()
    return None


def _bucket_for_entity_id(entity_id: str, *, bucket_count: int) -> int:
    """Stable bucket number for an entity id.

    We deliberately use a stable hash (md5) instead of Python's hash() because
    hash randomization would change partitions across processes.
    """

    import hashlib

    if bucket_count <= 0:
        raise ValueError("bucket_count must be > 0")
    h = hashlib.md5(entity_id.encode("utf-8")).digest()
    # first 8 bytes as unsigned int
    val = int.from_bytes(h[:8], "little", signed=False)
    return val % bucket_count


def _hf_glob_for_entity_id(
    dataset: str,
    *,
    entity_id: str,
    partitioning: str,
    bucket_count: int,
) -> str:
    entity_type = _extract_openalex_entity_type(entity_id)
    if entity_type is None:
        # Can't prune; fall back to scanning everything (slow). Still correct.
        return f"hf://datasets/{dataset}/**/*.parquet"

    if partitioning == "entity_type":
        return f"hf://datasets/{dataset}/entity_type={entity_type}/*.parquet"

    if partitioning == "entity_type_bucket":
        b = _bucket_for_entity_id(entity_id, bucket_count=bucket_count)
        return f"hf://datasets/{dataset}/entity_type={entity_type}/bucket={b}/*.parquet"

    raise ValueError(
        f"Unknown partitioning scheme. Expected 'entity_type' or 'entity_type_bucket'. Got: {partitioning!r}"
    )


def make_hf_entity_embedding_function(
    dataset: str,
    *,
    id_column: str = "entity_id",
    embedding_column: str = "vec",
    partitioning: str = "entity_type",
    bucket_count: int = 256,
):
    """Create an entity embedding function backed by DuckDB over HF parquet.

    Returns a callable (entity_id -> np.ndarray | None).
    """

    if not dataset or not isinstance(dataset, str):
        raise TypeError("dataset must be a non-empty string, e.g. 'tide/vecalex'")

    # Lazy import: duckdb is an optional dependency.
    try:
        import duckdb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "duckdb is required for HuggingFace parquet entity embeddings. Install with: pip install 'vecalex[hf]'"
        ) from e

    def _lookup(entity_id: str) -> np.ndarray | None:
        if not isinstance(entity_id, str) or not entity_id:
            return None

        glob = _hf_glob_for_entity_id(
            dataset,
            entity_id=entity_id,
            partitioning=partitioning,
            bucket_count=bucket_count,
        )

        # NOTE: For HF parquet this is still a remote scan, but constrained to a small
        # subset of files due to partition pruning.
        sql = f"SELECT {embedding_column} as vec FROM '{glob}' WHERE {id_column} = ? LIMIT 1"

        try:
            res = duckdb.sql(sql, params=[entity_id]).fetchone()
        except Exception:
            logger.exception("Failed HF embedding lookup for %s (glob=%s)", entity_id, glob)
            return None

        if not res:
            return None

        vec = np.asarray(res[0], dtype=float)
        # parquet may store lists as 1D already; just enforce shape
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        return vec

    return _lookup


@lru_cache(maxsize=8)
def get_hf_entity_embedding_function(
    dataset: str,
    *,
    partitioning: str = "entity_type",
    bucket_count: int = 256,
    id_column: str = "entity_id",
    embedding_column: str = "vec",
):
    """Cached factory to avoid recreating closures repeatedly."""

    return make_hf_entity_embedding_function(
        dataset,
        id_column=id_column,
        embedding_column=embedding_column,
        partitioning=partitioning,
        bucket_count=bucket_count,
    )
