"""Unit tests for the vecalex.scopes module."""

import numpy as np
import pyalex
import pytest

import vecalex


def test_vecalex_config_passthrough():
    """Verify that pyalex config options are set when setting them in the vecalex config."""
    new_max_retries = vecalex.config.max_retries + 2
    assert pyalex.config.max_retries != new_max_retries
    vecalex.config.max_retries = new_max_retries
    assert pyalex.config.max_retries == new_max_retries


@pytest.fixture
def redefined_entity_classes():
    return [
        vecalex.Author,
        vecalex.Concept,
        vecalex.Domain,
        vecalex.Field,
        vecalex.Funder,
        vecalex.Institution,
        vecalex.Subfield,
        vecalex.Work,
    ]


@pytest.fixture
def redefined_query_classes():
    return [
        vecalex.Authors,
        vecalex.Concepts,
        vecalex.Domains,
        vecalex.Fields,
        vecalex.Funders,
        vecalex.Institutions,
        vecalex.Journals,
        vecalex.People,
        vecalex.Subfields,
        vecalex.Works,
    ]


@pytest.fixture
def original_entity_classes():
    return [
        vecalex.Publisher,
    ]


@pytest.fixture
def original_query_classes():
    return [
        vecalex.Publishers,
    ]


def test_openalex_entity_classes_redefined(redefined_entity_classes, original_entity_classes):
    """Verify that vecalex OpenAlex classes correctly extend pyalex classes."""
    for redefined_class in redefined_entity_classes:
        class_name = redefined_class.__name__
        original_class = getattr(pyalex, class_name)
        assert redefined_class is not original_class, (
            f"VecAlex class {class_name} must be a redefinition of the PyAlex class of the same name"
        )
        assert issubclass(redefined_class, original_class), (
            f"VecAlex class {class_name} must be a subclass of the PyAlex class of the same name"
        )
    for original_class in original_entity_classes:
        class_name = original_class.__name__
        redefined_class = getattr(vecalex, class_name)
        assert redefined_class is original_class, f"VecAlex class {class_name} must not be redefined"


def test_openalex_query_classes_redefined(redefined_query_classes, original_query_classes):
    """Verify that vecalex OpenAlex query classes correctly extend pyalex classes."""
    for redefined_class in redefined_query_classes:
        class_name = redefined_class.__name__
        original_class = getattr(pyalex, class_name)
        assert redefined_class is not original_class, (
            f"VecAlex class {class_name} must be a redefinition of the PyAlex class of the same name"
        )
        assert issubclass(redefined_class, original_class), (
            f"VecAlex class {class_name} must be a subclass of the PyAlex class of the same name"
        )
        expected_resource_class_name = class_name[:-1]  # Remove plural 's'
        expected_resource_class = getattr(vecalex, expected_resource_class_name)
        assert redefined_class.resource_class is expected_resource_class, (
            f"VecAlex query class {class_name} must use VecAlex entity class {expected_resource_class_name} "
            "as its resource_class"
        )
    for original_class in original_query_classes:
        class_name = original_class.__name__
        redefined_class = getattr(vecalex, class_name)
        assert redefined_class is original_class, f"VecAlex class {class_name} must not be redefined"


def test_vecalex_entity_vec_property(redefined_entity_classes):
    """Verify that VecAlexEntity has a 'vec' property."""
    for entity_class in redefined_entity_classes:
        if entity_class is vecalex.Work:
            continue  # Work vec property tested separately
        entity = entity_class({"id": "https://openalex.org/A1234567890"})
        assert "works_api_url" not in entity and entity["vec"] is None, (
            "VecAlexEntity 'vec' property should be None when no `works_api_url` property is present"
        )


# mock the SentenceTransformer to avoid loading a real model
class MockSentenceTransformer:
    n_dims = 768

    def encode(self, docs, **kwargs):
        # return vectors of length 768 filled with zeros
        return np.full((len(docs), self.n_dims), fill_value=0.0)


vecalex.config.model = MockSentenceTransformer()


def test_vecalex_work_vec_property():
    """Verify that the `vec` property of a vecalex Work is computed from its abstract."""
    work = vecalex.Work(
        {
            "id": "https://openalex.org/W1234567890",
            "abstract_inverted_index": {"This": [0], "is": [1], "a": [2], "test": [3], "abstract.": [4]},
        }
    )

    assert work["vec"] is not None, "VecAlex Work 'vec' property should not be None when an abstract is present"
    assert len(work["vec"]) == MockSentenceTransformer.n_dims, (
        f"VecAlex Work 'vec' property should have length {MockSentenceTransformer.n_dims}"
    )
