"""Composition root: wires settings -> sources, embedder, vector store, LLM, and the graph.

The FastAPI app, the MCP server, the CLI, and the evals all go through this one class.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .agent import build_graph
from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .llm import LLM, build_llm
from .models import Document, Forecast, Market, ScoredChunk
from .polymarket import PolymarketClient
from .retrieval import Retriever
from .sources import Source, build_sources
from .sources.base import query_from_question
from .tracing import current_trace_id, flush, span, trace_url
from .vectorstore import VectorStore, build_store

log = logging.getLogger(__name__)


class ResearchService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        polymarket: PolymarketClient | None = None,
        sources: list[Source] | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        s = self.settings
        self.polymarket = polymarket or PolymarketClient(s.gamma_url, s.clob_url, s.http_timeout)
        self.sources = (
            sources
            if sources is not None
            else build_sources(s.source_names, timeout=s.http_timeout)
        )
        self.embedder = embedder or build_embedder(s)
        self.store = store or build_store(s, self.embedder.dim)
        self.retriever = Retriever(
            self.embedder,
            self.store,
            hybrid=s.hybrid,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
        )
        self.llm = llm or build_llm(s)
        self.graph = build_graph(self.retriever, self.llm, gather=self.ingest_market, top_k=s.top_k)

    # ---- markets -------------------------------------------------------
    def list_markets(self, limit: int = 20, active: bool = True) -> list[Market]:
        return self.polymarket.list_markets(limit=limit, active=active)

    def get_market(self, market_id: str) -> Market:
        return self.polymarket.get_market(market_id)

    # ---- ingestion -----------------------------------------------------
    def fetch_documents(self, market: Market, as_of: datetime | None = None) -> list[Document]:
        query = query_from_question(market.question)
        docs: list[Document] = []
        for src in self.sources:
            with span("fetch_source", as_type="tool", source=src.name, query=query) as sp:
                try:
                    got = src.fetch(query, as_of=as_of, limit=self.settings.max_docs_per_source)
                except Exception as exc:  # one flaky source should not kill the run
                    log.warning("source %s failed: %s", src.name, exc)
                    got = []
                sp.set("n_docs", len(got))
            docs.extend(got)
        seen: set[str] = set()
        unique = []
        for d in docs:
            key = d.title.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(d.model_copy(update={"market_id": market.id}))
        return unique

    def ingest_documents(self, market_id: str, docs: list[Document]) -> int:
        docs = [d.model_copy(update={"market_id": market_id}) for d in docs]
        return self.retriever.add_documents(docs)

    def ingest_market(self, market: Market, as_of: datetime | None = None) -> int:
        with span("ingest_market", as_type="span", market_id=market.id):
            return self.retriever.add_documents(self.fetch_documents(market, as_of))

    # ---- retrieval / research -----------------------------------------
    def search_evidence(
        self,
        market_id: str,
        query: str | None = None,
        k: int | None = None,
        market: Market | None = None,
    ) -> list[ScoredChunk]:
        market = market or self.get_market(market_id)
        if self.store.count(market.id) == 0:
            self.ingest_market(market)
        return self.retriever.search(
            query or market.question, k=k or self.settings.top_k, market_id=market.id
        )

    def research(
        self, market: Market, as_of: datetime | None = None, market_price: float | None = None
    ) -> Forecast:
        with span(
            "research_market",
            as_type="agent",
            input=market.question,
            market_id=market.id,
            llm=self.llm.name,
        ):
            state: dict[str, Any] = {"market": market, "as_of": as_of}
            if market_price is not None:
                state["market_price"] = market_price
            out = self.graph.invoke(state)
            tid = current_trace_id()
        flush()
        forecast: Forecast = out["forecast"]
        return forecast.model_copy(
            update={"llm": self.llm.name, "trace_id": tid, "trace_url": trace_url(tid)}
        )

    def research_market(self, market_id: str, as_of: datetime | None = None) -> Forecast:
        return self.research(self.get_market(market_id), as_of=as_of)
