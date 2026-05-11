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
      --bucket-count 256

Afterwards you can upload the `./partitioned` directory to HuggingFace Datasets
(using `huggingface-cli` / `git lfs`, or the `datasets` library).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


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
        "--bucket-count",
        type=int,
        default=256,
        help="Bucket count for partitioning (default: 256)",
    )
    p.add_argument(
        "--id-column",
        default="id",
        help="ID column name (default: id)",
    )
    p.add_argument(
        "--embedding-column",
        default="vec",
        help="Embedding column name (default: vec)",
    )
    p.add_argument("--memory", default=None, help="duckdb memory limit.")
    p.add_argument("--threads", default=None, help="duckdb threads.")
    p.add_argument("--cache", default=None, help="duckdb cache directory.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    if args.memory:
        con.execute("SET memory_limit=?", [args.memory])
    if args.threads:
        con.execute("SET threads=?", [args.threads])
    if args.cache:
        con.execute("SET temp_directory=?", [args.cache])

    query = f"""
    SELECT
        {args.id_column} as id,
        {args.embedding_column} as vec,
        regexp_extract({args.id_column}, '/([A-Z])\\d+$', 1) as entity_type,
        CAST(regexp_extract({args.id_column}, '[A-Z](\\d+)$', 1) AS UBIGINT) as entity_id,
        CAST(regexp_extract({args.id_column}, '[A-Z](\\d+)$', 1) AS UBIGINT) % {args.bucket_count} as bucket
    FROM read_parquet('{args.input_glob}')
    ORDER BY id
    """
    con.execute(
        f"""
        COPY ({query})
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (entity_type, bucket), COMPRESSION 'zstd');
        """
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
