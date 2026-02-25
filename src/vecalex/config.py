"""Configuration for vecalex.

vecalex configuration controls how we:
1) retrieve works (abstract sources) for an entity,
2) embed texts (abstracts),
3) aggregate embeddings into an entity vector,
4) optionally bypass work retrieval by loading precomputed entity vectors.

`pyalex` configuration (API key, mailto, etc.) should be set by users directly:

    import pyalex
    pyalex.config.api_key = "..."
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


WorkRetrievalFunction = Callable[[str], list[dict]]
EmbeddingFunction = Callable[[list[str]], np.ndarray]
AggregateEmbeddingsFunction = Callable[[np.ndarray], np.ndarray]
EntityEmbeddingFunction = Callable[[str], np.ndarray]


def _default_aggregate_embeddings(work_vectors: np.ndarray) -> np.ndarray:
    """Default aggregation: mean over works."""

    return work_vectors.mean(axis=0)


@dataclass
class VecAlexConfig:
    """vecalex runtime configuration.

    Notes
    -----
    - This config is a mutable singleton (see `config` below).
    - All callables are expected to be *pure-ish* (no mutation of inputs).
    """

    # Work retrieval
    max_works_per_entity: int = 20
    work_sorting: str = "publication_date:desc"
    work_retrieval_function: WorkRetrievalFunction | None = None

    # Embedding
    model_name_or_path: str = "EMBO/ModernBERT-neg-sampling-PubMed"
    embedding_function: EmbeddingFunction | None = None

    # Aggregation
    aggregate_embeddings: AggregateEmbeddingsFunction = field(default=_default_aggregate_embeddings)

    # Optional: precomputed entity vectors
    entity_embedding_function: EntityEmbeddingFunction | None = None


config = VecAlexConfig()
