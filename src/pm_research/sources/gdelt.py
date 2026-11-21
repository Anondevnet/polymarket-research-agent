"""GDELT 2.0 DOC API (free, no key): global news article search with date bounds."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from ..models import Document
from .base import before, doc_id, register_source

log = logging.getLogger(__name__)


@register_source("gdelt")
class GdeltSource:
    name = "gdelt"
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        timeout: float = 20.0,
        lookback_days: int = 30,
        client: httpx.Client | None = None,
        **_: Any,
    ) -> None:
        self.lookback_days = lookback_days
        self._http = client or httpx.Client(timeout=timeout, headers={"User-Agent": "pmra/0.1"})

    def params(self, query: str, as_of: datetime | None, limit: int) -> dict[str, Any]:
        p: dict[str, Any] = {
            "query": f"{query} sourcelang:english",
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(limit, 250),
            "sort": "datedesc",
        }
        if as_of is not None:
            start = as_of - timedelta(days=self.lookback_days)
            p["startdatetime"] = start.strftime("%Y%m%d%H%M%S")
            p["enddatetime"] = as_of.strftime("%Y%m%d%H%M%S")
        return p

    def fetch(
        self, query: str, *, as_of: datetime | None = None, limit: int = 25
    ) -> list[Document]:
        resp = self._http.get(self.url, params=self.params(query, as_of, limit))
        if resp.status_code == 429:  # GDELT rate-limits aggressively; degrade gracefully
            log.warning("GDELT rate limited; skipping")
            return []
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:  # GDELT returns plain-text errors for bad queries
            log.warning("GDELT returned non-JSON: %s", resp.text[:200])
            return []
        return self.parse(data, as_of=as_of, limit=limit)

    def parse(
        self, data: dict[str, Any], *, as_of: datetime | None = None, limit: int = 25
    ) -> list[Document]:
        docs: list[Document] = []
        for art in data.get("articles", []):
            published = None
            if art.get("seendate"):
                try:
                    published = datetime.strptime(art["seendate"], "%Y%m%dT%H%M%SZ").replace(
                        tzinfo=UTC
                    )
                except ValueError:
                    published = None
            if not before(published, as_of):
                continue
            title = str(art.get("title", "")).strip()
            docs.append(
                Document(
                    id=doc_id(self.name, art.get("url", ""), title),
                    source=f"{self.name}:{art.get('domain', '')}",
                    title=title,
                    text=title,
                    url=art.get("url", ""),
                    published_at=published,
                )
            )
            if len(docs) >= limit:
                break
        return docs
