from datetime import UTC, datetime

import httpx
import pytest
import respx

from pm_research.sources import GdeltSource, GoogleNewsSource, StaticSource, build_sources
from pm_research.sources.base import query_from_question

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Fed holds rates - Reuters</title><link>https://n/1</link>
<pubDate>Mon, 11 Aug 2025 12:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source>
<description>&lt;a href="x"&gt;Fed holds rates - Reuters&lt;/a&gt;</description></item>
<item><title>Fed cuts - AP</title><link>https://n/2</link>
<pubDate>Thu, 18 Sep 2025 12:00:00 GMT</pubDate></item>
<item><title>Undated item</title><link>https://n/3</link></item>
</channel></rss>"""

CUTOFF = datetime(2025, 9, 1, tzinfo=UTC)


def test_query_from_question() -> None:
    q = query_from_question("Will the Fed cut rates in September 2025?")
    assert q == "Fed cut rates September 2025"


def test_google_news_parse_and_lookahead_guard() -> None:
    src = GoogleNewsSource()
    all_docs = src.parse(RSS)
    assert len(all_docs) == 3
    guarded = src.parse(RSS, as_of=CUTOFF)
    assert [d.title for d in guarded] == ["Fed holds rates - Reuters"]
    assert guarded[0].source == "google_news:Reuters"
    assert "before:2025-09-01" in src.build_query("fed", CUTOFF)


@respx.mock
def test_google_news_fetch() -> None:
    respx.get(GoogleNewsSource.url).mock(return_value=httpx.Response(200, text=RSS))
    assert len(GoogleNewsSource().fetch("fed", limit=2)) == 2


def test_gdelt_parse_and_params() -> None:
    src = GdeltSource()
    data = {
        "articles": [
            {"title": "A", "url": "u1", "seendate": "20250810T120000Z", "domain": "x.com"},
            {"title": "B", "url": "u2", "seendate": "20250920T120000Z", "domain": "y.com"},
        ]
    }
    docs = src.parse(data, as_of=CUTOFF)
    assert [d.title for d in docs] == ["A"]
    p = src.params("fed", CUTOFF, 10)
    assert p["enddatetime"] == "20250901000000"


@respx.mock
def test_gdelt_degrades_on_rate_limit() -> None:
    respx.get(GdeltSource.url).mock(return_value=httpx.Response(429))
    assert GdeltSource().fetch("fed") == []


def test_registry() -> None:
    assert isinstance(build_sources(["static"])[0], StaticSource)
    with pytest.raises(ValueError):
        build_sources(["nope"])
