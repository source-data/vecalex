"""CLI: create a Hive-partitioned parquet dataset for fast entity-id lookups.

See vecalex.hf_embeddings.partition_entity_embeddings for details.

Usage
-----
    python scripts/partition_hf_entity_embeddings.py \
      --input-glob './raw_parquet/*.parquet' \
      --output-dir './partitioned' \
      --bucket-count 5000
"""

from __future__ import annotations

import argparse

from vecalex.hf_embeddings import partition_entity_embeddings


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-glob", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bucket-count", type=int, default=256)
    p.add_argument("--id-column", default="id")
    p.add_argument("--embedding-column", default="vec")
    p.add_argument("--memory", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--cache", default=None)
    args = p.parse_args()

    partition_entity_embeddings(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        bucket_count=args.bucket_count,
        id_column=args.id_column,
        embedding_column=args.embedding_column,
        memory=args.memory,
        threads=args.threads,
        cache=args.cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
