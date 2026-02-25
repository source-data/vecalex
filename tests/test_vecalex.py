"""Unit tests for vecalex Scope + config.

These tests intentionally avoid hitting the OpenAlex API.
"""

from __future__ import annotations

import numpy as np
import pytest

import vecalex
from vecalex.config import VecAlexConfig
from vecalex.scope import Scope


class MockEmbedder:
    """Deterministic embedder used to avoid sentence-transformers in tests."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def __call__(self, texts: list[str]) -> np.ndarray:
        # Encode each text into a vector where the first component is the length.
        # Then normalize to unit length so cosine similarities are stable.
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=float)
            v[0] = float(len(t))
            v[1] = float(sum(c.isalpha() for c in t))
            v[2] = float(sum(c.isdigit() for c in t))
            v[3] = 1.0
            v = v / (np.linalg.norm(v) + 1e-12)
            vecs.append(v)
        return np.vstack(vecs)


@pytest.fixture
def cfg() -> VecAlexConfig:
    cfg = VecAlexConfig()
    cfg.embedding_function = MockEmbedder(dim=4)
    # keep defaults for aggregation
    return cfg


def test_scope_from_single_text(cfg: VecAlexConfig):
    s = Scope("hello", cfg=cfg)
    assert len(s) == 1
    assert s.vectors.shape == (1, 4)


def test_scope_from_list_of_texts(cfg: VecAlexConfig):
    s = Scope(["a", "bb", "ccc"], cfg=cfg)
    assert len(s) == 3
    assert s.vectors.shape == (3, 4)


def test_similarities_shape(cfg: VecAlexConfig):
    a = Scope(["a", "bb"], cfg=cfg)
    b = Scope(["a", "cccc", "dd"], cfg=cfg)
    sims = a.similarities(b)
    assert sims.shape == (2, 3)


def test_closest_requires_single_item_scope(cfg: VecAlexConfig):
    s = Scope(["a", "bb"], cfg=cfg)
    with pytest.raises(ValueError):
        _ = s.closest(["a", "bb"], top_n=1)


def test_closest_sorts_by_similarity(cfg: VecAlexConfig):
    query = Scope("abc", cfg=cfg)
    candidates = ["a", "abc", "abcdef"]
    closest, sims = query.closest(candidates, top_n=2)
    assert len(closest) == 2
    assert sims.shape == (2,)
    # "abc" should be the closest to itself
    assert closest[0] == "abc"
    assert sims[0] >= sims[1]


def test_scope_entity_embedding_function_shortcut(cfg: VecAlexConfig):
    cfg.entity_embedding_function = lambda entity_id: np.array([1.0, 0.0, 0.0, 0.0])
    s = Scope({"id": "https://openalex.org/A123"}, cfg=cfg)
    assert len(s) == 1
    assert s.vectors.shape == (1, 4)
    assert np.allclose(s.vectors[0], np.array([1.0, 0.0, 0.0, 0.0]))


def test_scope_entity_work_retrieval_and_aggregation(cfg: VecAlexConfig):
    # Provide works with abstracts, embed them (mock), aggregate (mean).
    cfg.work_retrieval_function = lambda entity_id: [
        {"id": "https://openalex.org/W1", "abstract": "aaaa"},
        {"id": "https://openalex.org/W2", "abstract": "bbbbbb"},
    ]

    s = Scope({"id": "https://openalex.org/A123"}, cfg=cfg)
    assert len(s) == 1
    assert s.vectors.shape == (1, 4)

    # sanity: mean of two encoded vectors equals produced entity vector
    assert cfg.embedding_function is not None
    expected = cfg.embedding_function(["aaaa", "bbbbbb"]).mean(axis=0)
    assert np.allclose(s.vectors[0], expected)


def test_public_api_exports():
    assert hasattr(vecalex, "Scope")
    assert hasattr(vecalex, "config")
