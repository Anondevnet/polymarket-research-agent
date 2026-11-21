from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..models import Document

_STOP = {
    "will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "is", "for", "at", "and",
    "or", "before", "after", "end", "than", "more", "less", "what", "who", "which", "does",
    "do", "win", "reach", "hit", "above", "below", "between", "with", "as", "any", "this",
}  # fmt: skip


def query_from_question(question: str, max_terms: int = 8) -> str:
    """Turn a market question into a compact keyword search query."""
    q = re.sub(r"\?|\"|'s\b", "", question)
    terms = [t for t in re.findall(r"[A-Za-z0-9$%.\-]+", q) if t.lower() not in _STOP]
    return " ".join(terms[:max_terms])


def doc_id(source: str, url: str, title: str) -> str:
    return hashlib.sha1(f"{source}|{url}|{title}".encode()).hexdigest()[:16]


@runtime_checkable
class Source(Protocol):
    name: str

    def fetch(
        self, query: str, *, as_of: datetime | None = None, limit: int = 25
    ) -> list[Document]:
        """Return documents matching ``query`` published strictly before ``as_of`` if given."""
        ...


SourceFactory = Callable[..., Source]
_REGISTRY: dict[str, SourceFactory] = {}


def register_source(name: str) -> Callable[[SourceFactory], SourceFactory]:
    def deco(factory: SourceFactory) -> SourceFactory:
        _REGISTRY[name] = factory
        return factory

    return deco


def available_sources() -> list[str]:
    return sorted(_REGISTRY)


def build_sources(names: list[str], **kwargs: object) -> list[Source]:
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unknown sources {unknown}; available: {available_sources()}")
    return [_REGISTRY[n](**kwargs) for n in names]


def before(published: datetime | None, as_of: datetime | None) -> bool:
    """Look-ahead guard: drop undated docs and anything published at/after the cutoff."""
    if as_of is None:
        return True
    return published is not None and published < as_of
