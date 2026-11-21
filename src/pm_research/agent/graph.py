"""LangGraph research agent.

    gather -> retrieve -> summarize_evidence -> estimate_probability -> compare_to_market

The estimate node never sees the market price, so the agent's number is an independent
forecast; the final node computes the edge versus the market. Read-only: no trading.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..llm import LLM
from ..models import EvidenceSummary, Forecast, Market, ScoredChunk
from ..retrieval import Retriever
from ..sources.base import query_from_question
from ..tracing import span


class ResearchState(TypedDict, total=False):
    market: Market
    as_of: datetime | None
    market_price: float | None
    chunks: list[ScoredChunk]
    evidence: EvidenceSummary
    probability: float
    rationale: str
    forecast: Forecast
    ingested: int


Gatherer = Callable[[Market, datetime | None], int]


def build_graph(
    retriever: Retriever, llm: LLM, gather: Gatherer | None = None, top_k: int = 8
) -> Any:
    def gather_node(state: ResearchState) -> dict[str, Any]:
        market = state["market"]
        with span("gather", as_type="tool", market_id=market.id) as s:
            n = 0
            if gather is not None and retriever.store.count(market.id) == 0:
                n = gather(market, state.get("as_of"))
            s.set("ingested_chunks", n)
            return {"ingested": n}

    def retrieve_node(state: ResearchState) -> dict[str, Any]:
        market = state["market"]
        queries = [market.question, query_from_question(market.question)]
        results: dict[str, ScoredChunk] = {}
        for q in queries:
            for sc in retriever.search(q, k=top_k, market_id=market.id):
                if sc.chunk.id not in results or sc.score > results[sc.chunk.id].score:
                    results[sc.chunk.id] = sc
        as_of = state.get("as_of")
        ranked = sorted(results.values(), key=lambda s: -s.score)
        if as_of is not None:  # second look-ahead guard at read time
            ranked = [r for r in ranked if r.chunk.published_at and r.chunk.published_at < as_of]
        return {"chunks": ranked[:top_k]}

    def summarize_node(state: ResearchState) -> dict[str, Any]:
        with span("summarize_evidence", as_type="chain") as s:
            ev = llm.summarize(state["market"], state.get("chunks", []))
            s.output(ev.model_dump())
            return {"evidence": ev}

    def estimate_node(state: ResearchState) -> dict[str, Any]:
        with span("estimate_probability", as_type="chain") as s:
            p, why = llm.estimate(state["market"], state["evidence"], state.get("as_of"))
            s.output({"probability": p, "rationale": why})
            return {"probability": p, "rationale": why}

    def compare_node(state: ResearchState) -> dict[str, Any]:
        market = state["market"]
        price = state.get("market_price")
        if price is None:
            price = market.yes_price
        p = state["probability"]
        forecast = Forecast(
            market_id=market.id,
            question=market.question,
            probability=round(p, 4),
            rationale=state.get("rationale", ""),
            market_price=price,
            edge=round(p - price, 4) if price is not None else None,
            evidence=state["evidence"],
            citations=list(
                dict.fromkeys(c.chunk.url for c in state.get("chunks", []) if c.chunk.url)
            ),
            as_of=state.get("as_of"),
        )
        return {"forecast": forecast}

    g: StateGraph[ResearchState] = StateGraph(ResearchState)
    g.add_node("gather", gather_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("summarize_evidence", summarize_node)
    g.add_node("estimate_probability", estimate_node)
    g.add_node("compare_to_market", compare_node)
    g.add_edge(START, "gather")
    g.add_edge("gather", "retrieve")
    g.add_edge("retrieve", "summarize_evidence")
    g.add_edge("summarize_evidence", "estimate_probability")
    g.add_edge("estimate_probability", "compare_to_market")
    g.add_edge("compare_to_market", END)
    return g.compile()
