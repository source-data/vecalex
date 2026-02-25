"""vecalex public API.

The library no longer subclasses / augments `pyalex` entity types.
Import entities from `pyalex` directly and use :class:`vecalex.Scope` to embed
and compare texts / entities.
"""

from vecalex.config import config
from vecalex.scope import Scope

__all__ = ["Scope", "config"]
