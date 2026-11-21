from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from pm_research.config import Settings
from pm_research.embeddings import HashingEmbedder
from pm_research.models import Document, Market
from pm_research.service import ResearchService
from pm_research.sources import StaticSource
from pm_research.vectorstore import InMemoryVectorStore


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


MARKET = Market(
    id="m1",
    question="Will the Federal Reserve cut interest rates in September 2025?",
    description="Resolves YES if the FOMC lowers the target range at its September meeting.",
    end_date=dt("2025-09-17T18:00:00"),
    yes_price=0.62,
    yes_token_id="tok-yes",
)

DOCS = [
    Document(
        id="d1",
        source="static",
        title="Powell signals rate cut likely in September",
        text="Fed Chair Powell said the time has come to adjust policy; a cut is expected.",
        published_at=dt("2025-08-22T15:00:00"),
    ),
    Document(
        id="d2",
        source="static",
        title="Hot inflation print dents rate cut hopes",
        text="Consumer prices rose more than expected, and traders doubt the Fed will cut.",
        published_at=dt("2025-08-12T13:00:00"),
    ),
    Document(
        id="d3",
        source="static",
        title="Fed cuts rates by a quarter point",
        text="The Federal Reserve lowered rates in September.",
        published_at=dt("2025-09-17T19:00:00"),
    ),  # after cutoff: must never be used
    Document(
        id="d4",
        source="static",
        title="Bitcoin rallies to record high",
        text="Crypto markets surged as bitcoin set a record.",
        published_at=dt("2025-08-20T10:00:00"),
    ),
]


class FakePolymarket:
    def __init__(self, markets: list[Market] | None = None) -> None:
        self.markets = {m.id: m for m in (markets or [MARKET])}

    def list_markets(self, limit: int = 20, active: bool = True) -> list[Market]:
        return list(self.markets.values())[:limit]

    def get_market(self, market_id: str) -> Market:
        if market_id not in self.markets:
            raise ValueError(f"unknown market {market_id}")
        return self.markets[market_id]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        vector_store="memory",
        embedding_provider="hashing",
        llm_provider="heuristic",
        sources="static",
        top_k=5,
    )


@pytest.fixture
def service(settings: Settings) -> ResearchService:
    return ResearchService(
        settings,
        polymarket=FakePolymarket(),  # type: ignore[arg-type]
        sources=[StaticSource(DOCS)],
        embedder=HashingEmbedder(256),
        store=InMemoryVectorStore(),
    )


class FakeOpenAI:
    """Mimics ``client.chat.completions.create`` returning canned JSON."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Completions:
            def create(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                content = outer.replies.pop(0)

                class _Msg:
                    pass

                msg = _Msg()
                msg.content = content  # type: ignore[attr-defined]
                choice = type("C", (), {"message": msg})()
                usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
                return type("R", (), {"choices": [choice], "usage": usage})()

        self.chat = type("Chat", (), {"completions": _Completions()})()
