"""Scopes class extending the base pyalex classes."""

from functools import cached_property
from logging import getLogger
from typing import Any

import pyalex
import pyalex.api
from numpy import ndarray

logger = getLogger(__name__)


class VecAlexConfig(pyalex.api.AlexConfig):
    """
    Configuration class for VecAlex.

    All pyalex configuration options can be set here. In addition:

    Attributes
    ----------
    model : SentenceTransformer | None
        The embedding model instance. If None, it will be initialized on first use with the specified model_name.
    model_name : str | None
        Name of the embedding model to use. If None, defaults to "all-MiniLM-L6-v2".
    n_works_max : int
        Maximum number of works to consider when computing entity vectors. Defaults to 100.
    """

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, value):
        # pass through to pyalex's config
        pyalex.config[key] = value
        return super().__setitem__(key, value)


config = VecAlexConfig(
    model=None,
    model_name="all-MiniLM-L6-v2",
    n_works_max=100,
    **pyalex.config,
)


def _works_for_entity(resource: pyalex.api.OpenAlexEntity) -> list[pyalex.Work]:
    """
    Fetch works associated with a given entity that have an abstract.

    Parameters
    ----------
    resource : OpenAlexEntity
        The OpenAlex entity for which to fetch associated works.

    Returns
    -------
    list[Work]
        A list of associated works (only id and abstract/abstract_inverted_index fields).
    """
    assert 0 < config.n_works_max <= 10000, "`n_works_max` must be between 1 and 10000."
    resource_id = resource["id"]

    if isinstance(resource, pyalex.Work):
        logger.debug("Resource %s is a work itself, using its own abstract.", resource_id)
        return [resource]

    # if the resource has no works_api_url, return an empty list. this is the case at least for publishers, who
    # have a sources_api_url instead.
    if "works_api_url" not in resource:
        logger.debug("Resource %s has no `works_api_url` field, cannot fetch works.", resource_id)
        return []

    works_api_url = resource["works_api_url"]
    works_api_url += f",has_abstract:true&per-page={config.n_works_max}&select=id,abstract_inverted_index"
    logger.debug("Fetching works for %s from %s", resource_id, works_api_url)
    works = pyalex.Works()._get_from_url(works_api_url)
    return works


def _compute_vec(documents: list[str]) -> ndarray | None:
    """
    Compute the vector for a given list of documents.

    Parameters
    ----------
    documents : list[str]
        The list of documents to compute the vector for.

    Returns
    -------
    ndarray | None
        The computed vector or None if no documents are provided.
    """
    logger.debug(f"Computing vector for {len(documents)} documents.")
    documents = list(filter(None, documents))  # remove empty documents
    if not documents:
        return None
    model = _init_embedding_model()
    vectors = model.encode(documents, show_progress_bar=False, normalize_embeddings=True)
    return vectors.mean(axis=0)


def _init_embedding_model() -> Any:
    """
    Initialize the embedding model.

    Returns
    -------
    SentenceTransformer
        The initialized embedding model.
    """
    assert config.model or config.model_name, "Either config.model or config.model_name must be set."
    if config.model is None:
        logger.debug(f"Initializing embedding model: {config.model_name}")
        from sentence_transformers import SentenceTransformer

        config.model = SentenceTransformer(config.model_name)
    return config.model


class VecAlexEntity(pyalex.api.OpenAlexEntity):
    """
    Base class for VecAlex entities with vector computation capability.

    Extends the pyalex OpenAlexEntity class to add a `vec` property that computes
    the vector representation of the entity based on its own abstract or its associated works' abstracts.
    """

    def __getitem__(self, key):
        if key == "vec":
            return self._vec
        return super().__getitem__(key)

    @cached_property
    def _vec(self) -> ndarray | None:
        """
        Compute the vector for this entity.

        Returns
        -------
        ndarray | None
            The computed vector or None if no documents are provided.
        """
        works = _works_for_entity(self)
        abstracts = [work["abstract"] for work in works]
        return _compute_vec(abstracts)


##############################################################################################
# Below: redefinitions of OpenAlex classes to use VecAlexEntity as base class
##############################################################################################


class Author(VecAlexEntity, pyalex.Author):
    pass


class Authors(pyalex.Authors):
    resource_class = Author


class Concept(VecAlexEntity, pyalex.Concept):
    pass


class Concepts(pyalex.Concepts):
    resource_class = Concept


class Domain(VecAlexEntity, pyalex.Domain):
    pass


class Domains(pyalex.Domains):
    resource_class = Domain


class Field(VecAlexEntity, pyalex.Field):
    pass


class Fields(pyalex.Fields):
    resource_class = Field


class Funder(VecAlexEntity, pyalex.Funder):
    pass


class Funders(pyalex.Funders):
    resource_class = Funder


class Institution(VecAlexEntity, pyalex.Institution):
    pass


class Institutions(pyalex.Institutions):
    resource_class = Institution


class Subfield(VecAlexEntity, pyalex.Subfield):
    pass


class Subfields(pyalex.Subfields):
    resource_class = Subfield


class Source(VecAlexEntity, pyalex.Source):
    pass


class Sources(pyalex.Sources):
    resource_class = Source


class Topic(VecAlexEntity, pyalex.Topic):
    pass


class Topics(pyalex.Topics):
    resource_class = Topic


class Work(VecAlexEntity, pyalex.Work):
    pass


class Works(pyalex.Works):
    resource_class = Work


# aliases
People = Authors
Journals = Sources
