"""In-memory / JSONL source used for tests, offline evals, and bring-your-own-data."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import Document
from .base import before, register_source


@register_source("static")
class StaticSource:
    name = "static"

    def __init__(
        self, documents: list[Document] | None = None, path: str | Path | None = None, **_: Any
    ) -> None:
        self.documents: list[Document] = list(documents or [])
        if path:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    self.documents.append(Document.model_validate(json.loads(line)))

    def fetch(
        self, query: str, *, as_of: datetime | None = None, limit: int = 25
    ) -> list[Document]:
        terms = {t.lower() for t in re.findall(r"\w+", query) if len(t) > 2}
        scored = []
        for d in self.documents:
            if not before(d.published_at, as_of):
                continue
            words = set(re.findall(r"\w+", f"{d.title} {d.text}".lower()))
            overlap = len(terms & words)
            if overlap:
                scored.append((overlap, d))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:limit]]
