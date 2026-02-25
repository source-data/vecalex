"""Scope: embed texts and OpenAlex entities, then compare them.

vecalex intentionally does **not** augment/modify pyalex entity types.
Instead, users create a :class:`Scope` from:

- a text (str) or list of texts (list[str])
- an OpenAlex entity dict (must include an "id") or list of such dicts

and then call:

- :meth:`Scope.closest` for top-N most similar items (single-item scopes only)
- :meth:`Scope.similarities` for pairwise cosine similarities

Work retrieval and embedding are customizable via :mod:`vecalex.config`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import numpy as np
from pyalex import invert_abstract, Works
from tqdm import tqdm
from vecalex.config import VecAlexConfig, config

logger = getLogger(__name__)


OpenAlexEntityLike = dict[str, Any]
ScopeItem = str | OpenAlexEntityLike


def _is_entity_like(obj: Any) -> bool:
    return isinstance(obj, dict) and "id" in obj


def _normalize_items(value: Any) -> list[ScopeItem]:
    """Normalize supported constructor inputs into a list of scope items."""

    if isinstance(value, Scope):
        return list(value.items)
    if isinstance(value, str) or _is_entity_like(value):
        return [value]
    if isinstance(value, Sequence):
        items: list[ScopeItem] = []
        for v in value:
            if not (isinstance(v, str) or _is_entity_like(v)):
                raise TypeError(
                    "Scope input sequences must contain only strings or entity dicts with an 'id' key. "
                    f"Got element type: {type(v)!r}"
                )
            items.append(v)
        return items
    raise TypeError(
        "Unsupported Scope input. Expected str, entity dict, list[str], list[entity dict], or Scope. "
        f"Got: {type(value)!r}"
    )


def _init_default_embedding_function(cfg: VecAlexConfig):
    """Lazy-init sentence-transformers model and return an embedding callable."""

    from sentence_transformers import SentenceTransformer  # expensive import, so we do it lazily only if needed

    model = SentenceTransformer(cfg.model_name_or_path)

    def _encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    return _encode


def _embed_texts(texts: list[str], *, cfg: VecAlexConfig) -> np.ndarray:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return np.empty((0, 0), dtype=float)

    embed = cfg.embedding_function
    if embed is None:
        embed = _init_default_embedding_function(cfg)
    vectors = np.asarray(embed(texts))

    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(
            "embedding_function must return a 2D numpy array of shape (len(texts), embedding_dim). "
            f"Got {vectors.shape} for len(texts)={len(texts)}."
        )
    return vectors


def _extract_work_abstract(work: dict[str, Any]) -> str | None:
    if work.get("abstract"):
        return work["abstract"]
    if work.get("abstract_inverted_index"):
        try:
            

            return invert_abstract(work["abstract_inverted_index"])
        except Exception:
            logger.exception("Failed to invert abstract_inverted_index")
            return None
    return None


def _default_work_retrieval_function(entity: OpenAlexEntityLike, *, cfg: VecAlexConfig) -> list[dict[str, Any]]:
    """Default: use entity['works_api_url'] if present.

    This matches the old vecalex behavior but **without** subclassing pyalex.

    Returns a list of works containing at least abstract / abstract_inverted_index.
    """
    if not (1 <= cfg.max_works_per_entity <= 10000):
        raise ValueError("config.max_works_per_entity must be between 1 and 10000")

    # Special case: the entity is actually a work.
    entity_id = entity.get("id")
    if isinstance(entity_id, str) and entity_id.rsplit("/", 1)[-1].startswith("W"):
        return [entity]

    works_api_url = entity.get("works_api_url")
    if not works_api_url:
        return []

    # works_api_url already contains filters; we append additional constraints.
    # OpenAlex uses `per-page`, and pyalex supports internal _get_from_url.
    url = (
        f"{works_api_url},has_abstract:true"
        f"&per-page={cfg.max_works_per_entity}"
        f"&sort={cfg.work_sorting}"
        f"&select=id,abstract_inverted_index"
    )
    logger.debug("Fetching works for %s from %s", entity_id, url)
    return Works()._get_from_url(url)


def _entity_vector(entity: OpenAlexEntityLike, *, cfg: VecAlexConfig) -> np.ndarray | None:
    entity_id = entity.get("id")
    if not isinstance(entity_id, str):
        raise TypeError("Entity dict must have an 'id' key with a string value")

    if cfg.entity_embedding_function is not None:
        vec = np.asarray(cfg.entity_embedding_function(entity_id))
        if vec.ndim != 1:
            raise ValueError("entity_embedding_function must return a 1D vector")
        return vec

    work_retrieval = cfg.work_retrieval_function
    if work_retrieval is None:
        works = _default_work_retrieval_function(entity, cfg=cfg)
    else:
        works = work_retrieval(entity_id)

    abstracts = [a for a in (_extract_work_abstract(w) for w in works) if a]
    if not abstracts:
        return None

    work_vectors = _embed_texts(abstracts, cfg=cfg)
    if work_vectors.size == 0:
        return None
    vec = np.asarray(cfg.aggregate_embeddings(work_vectors))
    if vec.ndim != 1:
        raise ValueError("aggregate_embeddings must return a 1D vector")
    return vec


def _item_vector(item: ScopeItem, *, cfg: VecAlexConfig) -> np.ndarray | None:
    if isinstance(item, str):
        vectors = _embed_texts([item], cfg=cfg)
        return None if vectors.size == 0 else vectors[0]
    if _is_entity_like(item):
        return _entity_vector(item, cfg=cfg)
    raise TypeError(f"Unsupported scope item type: {type(item)!r}")


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for two 2D arrays.

    If embeddings are already L2-normalized, this reduces to dot-product.
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected 2D arrays")
    # normalize defensively
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


@dataclass(frozen=True)
class Scope:
    """A set of items with cached vectors."""

    items: tuple[ScopeItem, ...]
    vectors: np.ndarray
    cfg: VecAlexConfig

    def __init__(self, value: Any, *, cfg: VecAlexConfig = config, progress: bool = True):
        object.__setattr__(self, "cfg", cfg)
        items = _normalize_items(value)
        vectors_list: list[np.ndarray] = []
        kept_items: list[ScopeItem] = []
        for item in tqdm(items, desc="Processing Scope items", unit="item", disable=not progress):
            vec = _item_vector(item, cfg=cfg)
            if vec is None:
                continue
            kept_items.append(item)
            vectors_list.append(np.asarray(vec, dtype=float))

        object.__setattr__(self, "items", tuple(kept_items))
        if not vectors_list:
            object.__setattr__(self, "vectors", np.empty((0, 0), dtype=float))
        else:
            mat = np.vstack([v.reshape(1, -1) for v in vectors_list])
            object.__setattr__(self, "vectors", mat)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.items)

    def similarities(self, others: Any) -> np.ndarray:
        other_scope = others if isinstance(others, Scope) else Scope(others, cfg=self.cfg)
        if self.vectors.size == 0 or other_scope.vectors.size == 0:
            return np.empty((len(self), len(other_scope)), dtype=float)
        return _cosine_similarity_matrix(self.vectors, other_scope.vectors)

    def closest(self, others: Any, *, top_n: int = 5) -> tuple[list[ScopeItem], np.ndarray]:
        if len(self) != 1:
            raise ValueError("Scope.closest is only supported for single-item scopes (len(self) == 1)")

        other_scope = others if isinstance(others, Scope) else Scope(others, cfg=self.cfg)
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        if self.vectors.size == 0 or other_scope.vectors.size == 0 or len(other_scope) == 0:
            return [], np.asarray([], dtype=float)

        sims = _cosine_similarity_matrix(self.vectors, other_scope.vectors).reshape(-1)
        order = np.argsort(sims)[::-1]
        order = order[: min(top_n, len(order))]
        closest_items = [other_scope.items[i] for i in order]
        return closest_items, sims[order]
