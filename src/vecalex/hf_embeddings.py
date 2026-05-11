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


def make_hf_entity_embedding_function(dataset: str):
    """Create an entity embedding function backed by DuckDB over HF parquet.

    Returns a callable (entity_id -> np.ndarray).
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

    glob = f"hf://datasets/{dataset}/**/*.parquet"
    res_bucket_count = duckdb.sql("SELECT MAX(bucket) + 1 FROM read_parquet($glob)", params={"glob": glob}).fetchone()
    if res_bucket_count is None or res_bucket_count[0] is None:
        raise ValueError(
            f"Failed to determine bucket_count from dataset '{dataset}'. Ensure it is partitioned correctly."
        )
    bucket_count = res_bucket_count[0]

    def _lookup(entity_id: str) -> np.ndarray:
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("entity_id must be a non-empty string")

        # NOTE: For HF parquet this is still a remote scan, but constrained to a small
        # subset of files due to partition pruning.
        query = """
        SELECT vec
        FROM read_parquet($glob)
        WHERE
            id = $id
            AND entity_type = regexp_extract($id, '/([A-Z])\\d+$', 1)
            AND bucket = CAST(regexp_extract($id, '[A-Z](\\d+)$', 1) AS UBIGINT) % $bucket_count
        """

        res = duckdb.sql(query, params={"glob": glob, "id": entity_id, "bucket_count": bucket_count}).fetchone()
        if not res:
            raise ValueError(f"Entity ID {entity_id} not found in {glob}")

        vec = np.asarray(res[0], dtype=float)
        # parquet may store lists as 1D already; just enforce shape
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        return vec

    return _lookup


@lru_cache(maxsize=8)
def get_hf_entity_embedding_function(dataset: str):
    """Cached factory to avoid recreating closures repeatedly."""

    return make_hf_entity_embedding_function(dataset)
