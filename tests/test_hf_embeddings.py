"""Tests for vecalex.hf_embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vecalex.hf_embeddings import _make_lookup, partition_entity_embeddings

duckdb = pytest.importorskip("duckdb")


BUCKET_COUNT = 5000

# ── Small-sample correctness tests ────────────────────────────────────────────

_B = 3  # bucket_count for correctness tests

# (id, vec, abstract)  —  bucket = numeric_part % _B
_SAMPLE = [
    ("https://openalex.org/W1", [1.0, 0.0, 0.0, 0.0], "aaaa"),  # W / bucket 1
    ("https://openalex.org/W2", [0.0, 1.0, 0.0, 0.0], "bbbb"),  # W / bucket 2
    ("https://openalex.org/W3", [0.0, 0.0, 1.0, 0.0], "cccc"),  # W / bucket 0
    ("https://openalex.org/W4", [0.5, 0.5, 0.0, 0.0], "dddd"),  # W / bucket 1 (with W1)
    ("https://openalex.org/A1", [0.0, 0.0, 0.0, 1.0], "eeee"),  # A / bucket 1
    ("https://openalex.org/A5", [0.3, 0.3, 0.3, 0.1], "ffff"),  # A / bucket 2
]

# Expected: partition path → set of IDs that should land there
_EXPECTED: dict[tuple[str, int], set[str]] = {
    ("A", 1): {"https://openalex.org/A1"},
    ("A", 2): {"https://openalex.org/A5"},
    ("W", 0): {"https://openalex.org/W3"},
    ("W", 1): {"https://openalex.org/W1", "https://openalex.org/W4"},
    ("W", 2): {"https://openalex.org/W2"},
}

_VEC = {id_: vec for id_, vec, _ in _SAMPLE}


@pytest.fixture(scope="module")
def small_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run partition_entity_embeddings once on _SAMPLE; return the output dir."""
    tmp = tmp_path_factory.mktemp("small")
    in_dir = tmp / "input"
    in_dir.mkdir()

    con = duckdb.connect()
    con.execute("CREATE TABLE t (id VARCHAR, vec FLOAT[4], abstract VARCHAR)")
    for id_, vec, abstract in _SAMPLE:
        vec_sql = f"[{', '.join(str(v) for v in vec)}]::FLOAT[4]"
        con.execute(f"INSERT INTO t VALUES (?, {vec_sql}, ?)", [id_, abstract])
    con.execute(f"COPY t TO '{in_dir / 'data.parquet'}' (FORMAT PARQUET)")
    con.close()

    partition_entity_embeddings(
        input_glob=str(in_dir / "*.parquet"),
        output_dir=tmp / "output",
        bucket_count=_B,
        threads=1,
    )
    return tmp / "output"


def test_partition_directories(small_out: Path) -> None:
    """Exactly the right entity_type/bucket leaf dirs are created."""
    leaf_dirs = {p.parent.relative_to(small_out) for p in small_out.rglob("*.parquet")}
    expected = {Path(f"entity_type={et}/bucket={b}") for et, b in _EXPECTED}
    assert leaf_dirs == expected


def test_one_file_per_partition_small(small_out: Path) -> None:
    """threads=1 + correct sort → exactly one file per leaf dir."""
    for leaf in {p.parent for p in small_out.rglob("*.parquet")}:
        files = list(leaf.glob("*.parquet"))
        assert len(files) == 1, f"{leaf.relative_to(small_out)} has {len(files)} files"


def test_columns_per_file(small_out: Path) -> None:
    """Only id and vec columns are kept from input parquet files."""
    for leaf in {p.parent for p in small_out.rglob("*.parquet")}:
        files = list(leaf.glob("*.parquet"))
        for file in files:
            con = duckdb.connect()
            columns = con.execute(f"DESCRIBE SELECT * FROM '{file}'").fetchall()
            con.close()
            column_names = {col[0] for col in columns}
            assert column_names == {"id", "vec", "entity_type", "bucket"}, (
                f"{file.relative_to(small_out)} has columns {column_names}"
            )


def test_row_counts(small_out: Path) -> None:
    """Each partition file contains the expected number of rows."""
    con = duckdb.connect()
    for (et, b), expected_ids in _EXPECTED.items():
        glob = str(small_out / f"entity_type={et}" / f"bucket={b}" / "*.parquet")
        (count,) = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()
        assert count == len(expected_ids), f"entity_type={et}/bucket={b}: got {count} rows"
    con.close()


def test_row_ids(small_out: Path) -> None:
    """Each partition contains exactly the right IDs."""
    con = duckdb.connect()
    for (et, b), expected_ids in _EXPECTED.items():
        glob = str(small_out / f"entity_type={et}" / f"bucket={b}" / "*.parquet")
        rows = con.execute(f"SELECT id FROM read_parquet('{glob}')").fetchall()
        assert {r[0] for r in rows} == expected_ids
    con.close()


def test_row_vectors(small_out: Path) -> None:
    """Vectors survive the parquet round-trip unchanged."""
    con = duckdb.connect()
    rows = con.execute(f"SELECT id, vec FROM read_parquet('{small_out}/**/*.parquet')").fetchall()
    con.close()
    assert len(rows) == len(_SAMPLE)
    for id_, vec in rows:
        assert list(vec) == pytest.approx(_VEC[id_], abs=1e-6)


def test_lookup_round_trip(small_out: Path) -> None:
    """_make_lookup on local partitioned output returns the correct vectors."""
    lookup = _make_lookup(str(small_out) + "/**/*.parquet")
    for id_, expected_vec, _ in _SAMPLE:
        result = lookup(id_)
        np.testing.assert_allclose(result, expected_vec, atol=1e-6)


def test_one_file_per_leaf_partition(tmp_path: Path) -> None:
    """Nearly all (entity_type, bucket) dirs must contain exactly one parquet file.

    With ORDER BY id (lexicographic), W1/W10/W100/W1000 interleave bucket
    assignments across DuckDB's parallel sort ranges, so almost every bucket
    receives files from both threads (~4650/5000 dirs with 2 files each).
    ORDER BY entity_type, bucket, id keeps partition data contiguous; at most a
    handful of buckets straddle the thread-split boundary (~4 dirs with 2 files).
    """
    in_dir = tmp_path / "input"
    in_dir.mkdir()

    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("""
        CREATE TABLE t AS
        SELECT 'https://openalex.org/W' || range::VARCHAR AS id,
               [random()::FLOAT for i in range(4)] AS vec
        FROM range(1, 1_000_001)
    """)
    con.execute(f"COPY t TO '{in_dir / 'data.parquet'}' (FORMAT PARQUET)")
    con.close()

    partition_entity_embeddings(
        input_glob=str(in_dir / "*.parquet"),
        output_dir=tmp_path / "output",
        bucket_count=BUCKET_COUNT,
        threads=2,
    )

    dirs_with_multiple = sum(
        1
        for leaf in set(p.parent for p in (tmp_path / "output").rglob("*.parquet"))
        if len(list(leaf.glob("*.parquet"))) > 1
    )
    # Correct sort: only buckets at thread-split boundaries get >1 file.
    # Wrong ORDER BY id: ~90% of buckets get >1 file.
    assert dirs_with_multiple <= BUCKET_COUNT // 100, (
        f"{dirs_with_multiple}/{BUCKET_COUNT} partition dirs have multiple files"
    )
