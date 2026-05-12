"""HuggingFace-hosted entity embeddings.

This module provides a lightweight retrieval path for *precomputed* entity vectors
stored as parquet on HuggingFace, and a helper to create the partitioned dataset.

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
from pathlib import Path

import numpy as np

logger = getLogger(__name__)


def _make_lookup(glob: str):
    """Return an (entity_id -> np.ndarray) callable for any parquet glob."""
    try:
        import duckdb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "duckdb is required for HuggingFace parquet entity embeddings. Install with: pip install 'vecalex[hf]'"
        ) from e

    res = duckdb.sql("SELECT MAX(bucket) + 1 FROM read_parquet($glob)", params={"glob": glob}).fetchone()
    if res is None or res[0] is None:
        raise ValueError(f"Failed to determine bucket_count from '{glob}'. Ensure it is partitioned correctly.")
    bucket_count = res[0]

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
            AND bucket = CAST(regexp_extract($id, '/[A-Z](\\d+)$', 1) AS UBIGINT) % $bucket_count
        """

        row = duckdb.sql(query, params={"glob": glob, "id": entity_id, "bucket_count": bucket_count}).fetchone()
        if not row:
            raise ValueError(f"Entity ID {entity_id} not found in {glob}")

        vec = np.asarray(row[0], dtype=float)
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        return vec

    return _lookup


def make_hf_entity_embedding_function(dataset: str):
    """Create an entity embedding function backed by DuckDB over HF parquet.

    Returns a callable (entity_id -> np.ndarray).
    """
    if not dataset or not isinstance(dataset, str):
        raise TypeError("dataset must be a non-empty string, e.g. 'tide/vecalex'")
    return _make_lookup(f"hf://datasets/{dataset}/**/*.parquet")


@lru_cache(maxsize=8)
def get_hf_entity_embedding_function(dataset: str):
    """Cached factory to avoid recreating closures repeatedly."""
    return make_hf_entity_embedding_function(dataset)


def partition_entity_embeddings(
    input_glob: str,
    output_dir: str | Path,
    *,
    bucket_count: int = 256,
    id_column: str = "id",
    embedding_column: str = "vec",
    memory: str | None = None,
    threads: int | None = None,
    cache: str | None = None,
    max_open_files: int | None = None,
) -> None:
    """Write a Hive-partitioned parquet dataset for fast entity-id lookups.

    Partitions by (entity_type, bucket). The ORDER BY clause sorts on partition
    columns first so every (entity_type, bucket) leaf receives a single
    contiguous write — preventing DuckDB from creating many tiny files per
    partition directory when rows are scattered across processing batches.
    """
    try:
        import duckdb  # type: ignore
    except ImportError as e:
        raise ImportError("duckdb is required. Install with: pip install 'vecalex[hf]'") from e

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    if memory:
        con.execute("SET memory_limit=?", [memory])
    if threads is not None:
        con.execute("SET threads=?", [threads])
    if cache:
        con.execute("SET temp_directory=?", [cache])
    if max_open_files is not None:
        con.execute("SET partitioned_write_max_open_files=?", [max_open_files])

    con.execute(
        """
        COPY (
            SELECT
                id,
                vec,
                regexp_extract(id, '/([A-Z])\\d+$', 1) AS entity_type,
                CAST(regexp_extract(id, '[A-Z](\\d+)$', 1) AS UBIGINT) % $bucket_count AS bucket
            FROM read_parquet($input_glob)
            ORDER BY entity_type, bucket, id
        )
        TO $output_dir (
            FORMAT PARQUET,
            PARTITION_BY (entity_type, bucket),
            COMPRESSION 'zstd'
        );
    """,
        {"input_glob": input_glob, "bucket_count": bucket_count, "output_dir": out_dir.as_posix()},
    )
