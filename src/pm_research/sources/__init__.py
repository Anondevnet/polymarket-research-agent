"""Pluggable evidence sources. Register new ones with :func:`register_source`."""

from .base import Source, available_sources, build_sources, register_source
from .gdelt import GdeltSource
from .google_news import GoogleNewsSource
from .static import StaticSource

__all__ = [
    "GdeltSource",
    "GoogleNewsSource",
    "Source",
    "StaticSource",
    "available_sources",
    "build_sources",
    "register_source",
]
