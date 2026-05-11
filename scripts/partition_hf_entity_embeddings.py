"""Create a Hive-partitioned parquet dataset for fast entity-id lookups.

Why this exists
--------------
When your entity embeddings live in many parquet files (especially remote ones on
HuggingFace), a lookup like:

    SELECT vec FROM 'hf://datasets/<ds>/**/*.parquet' WHERE entity_id = '...'

is slow because the engine must touch lots of parquet files.

If you instead write the dataset with Hive partitions (directories like
`entity_type=A/` or `entity_type=A/bucket=42/`), DuckDB can prune the scan to a
tiny subset of files.

This script shows one way to create such partitions from existing parquet files.

Requirements
------------
    pip install duckdb

Usage (local input)
-------------------
    python scripts/partition_hf_entity_embeddings.py \
      --input-glob './raw_parquet/*.parquet' \
      --output-dir './partitioned' \
      --partitioning entity_type

Or with additional hash buckets (recommended for very large datasets):

    python scripts/partition_hf_entity_embeddings.py \
      --input-glob './raw_parquet/*.parquet' \
      --output-dir './partitioned' \
      --partitioning entity_type_bucket \
      --bucket-count 256

Afterwards you can upload the `./partitioned` directory to HuggingFace Datasets
(using `huggingface-cli` / `git lfs`, or the `datasets` library).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from duckdb.sqltypes import INTEGER, VARCHAR


def _bucket(entity_id: str, bucket_count: int) -> int:
    h = hashlib.md5(entity_id.encode("utf-8")).digest()
    val = int.from_bytes(h[:8], "little", signed=False)
    return val % bucket_count


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-glob",
        required=True,
        help="Input parquet glob (local path), e.g. './raw/*.parquet'",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to write partitioned parquet dataset",
    )
    p.add_argument(
        "--partitioning",
        choices=["entity_type", "entity_type_bucket"],
        default="entity_type",
        help="Hive partition scheme to generate",
    )
    p.add_argument(
        "--bucket-count",
        type=int,
        default=256,
        help="Bucket count for entity_type_bucket partitioning (default: 256)",
    )
    p.add_argument(
        "--id-column",
        default="entity_id",
        help="ID column name (default: entity_id)",
    )
    p.add_argument(
        "--embedding-column",
        default="vec",
        help="Embedding column name (default: vec)",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import duckdb

    con = duckdb.connect(database=":memory:")

    # Register a stable bucket function that matches vecalex client code.
    if args.partitioning == "entity_type_bucket":
        con.create_function(
            "vecalex_bucket",
            lambda s: _bucket(s, args.bucket_count),
            [VARCHAR],
            INTEGER,
        )

    # Extract entity type from OpenAlex IDs like "https://openalex.org/S123".
    # We rely on the last path component starting with the type letter.
    entity_type_expr = f"upper(regexp_extract({args.id_column}, '/([A-Za-z])[^/]*$', 1))"

    if args.partitioning == "entity_type":
        query = f"""
        SELECT
            {args.id_column} as entity_id,
            {args.embedding_column} as vec,
            {entity_type_expr} as entity_type
        FROM read_parquet('{args.input_glob}')
        """
        con.execute(
            f"""
            COPY ({query})
            TO '{out_dir.as_posix()}'
            (FORMAT PARQUET, PARTITION_BY (entity_type));
            """
        )
        return 0

    # entity_type_bucket
    query = f"""
    SELECT
        {args.id_column} as entity_id,
        {args.embedding_column} as vec,
        {entity_type_expr} as entity_type,
        vecalex_bucket({args.id_column}) as bucket
    FROM read_parquet('{args.input_glob}')
    """
    con.execute(
        f"""
        COPY ({query})
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (entity_type, bucket));
        """
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
