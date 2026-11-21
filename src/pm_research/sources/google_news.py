"""Google News RSS search (free, no key). Supports ``before:``/``after:`` date operators,
which we use to prevent look-ahead leakage when back-testing resolved markets.

Note: Google's feed terms allow personal, non-commercial use only.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from ..models import Document
from .base import before, doc_id, register_source

_TAG = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return html.unescape(_TAG.sub(" ", text or "")).replace("\xa0", " ").strip()


@register_source("google_news")
class GoogleNewsSource:
    name = "google_news"
    url = "https://news.google.com/rss/search"

    def __init__(
        self,
        timeout: float = 20.0,
        lookback_days: int = 30,
        client: httpx.Client | None = None,
        **_: Any,
    ) -> None:
        self.lookback_days = lookback_days
        self._http = client or httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 pmra"}
        )

    def build_query(self, query: str, as_of: datetime | None) -> str:
        if as_of is None:
            return query
        start = (as_of - timedelta(days=self.lookback_days)).date().isoformat()
        return f"{query} after:{start} before:{as_of.date().isoformat()}"

    def fetch(
        self, query: str, *, as_of: datetime | None = None, limit: int = 25
    ) -> list[Document]:
        params = {"q": self.build_query(query, as_of), "hl": "en-US", "gl": "US", "ceid": "US:en"}
        resp = self._http.get(self.url, params=params)
        resp.raise_for_status()
        return self.parse(resp.text, as_of=as_of, limit=limit)

    def parse(self, xml: str, *, as_of: datetime | None = None, limit: int = 25) -> list[Document]:
        feed = feedparser.parse(xml)
        docs: list[Document] = []
        for entry in feed.entries:
            published: datetime | None = None
            if entry.get("published"):
                try:
                    published = parsedate_to_datetime(entry["published"]).astimezone(UTC)
                except (TypeError, ValueError):
                    published = None
            if not before(published, as_of):
                continue
            title = _clean(entry.get("title", ""))
            summary = _clean(entry.get("summary", ""))
            source = entry.get("source", {}).get("title", "") if entry.get("source") else ""
            # Google's <description> usually just repeats the headline + outlet name.
            base = title.rsplit(" - ", 1)[0]
            text = title if not summary or summary.startswith(base) else f"{title}. {summary}"
            docs.append(
                Document(
                    id=doc_id(self.name, entry.get("link", ""), title),
                    source=f"{self.name}:{source}" if source else self.name,
                    title=title,
                    text=text,
                    url=entry.get("link", ""),
                    published_at=published,
                )
            )
            if len(docs) >= limit:
                break
        return docs
