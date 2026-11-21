"""MCP server (official ``mcp`` Python SDK) exposing the research agent as tools.

Run over stdio for Claude Desktop / Cursor:  ``pmra mcp``
or over streamable HTTP:                     ``pmra mcp --transport streamable-http``
(The FastAPI app also mounts it at ``/mcp``.)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .service import ResearchService

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True)


def build_mcp(get_service: Callable[[], ResearchService]) -> MCPServer:
    mcp = MCPServer(
        name="pm-research-agent",
        instructions=(
            "Read-only research tools for Polymarket prediction markets. Use list_markets to "
            "find a market id, search_evidence to see retrieved news, and research_market for "
            "an evidence-backed probability vs. the market price. No trading is possible."
        ),
    )

    @mcp.tool(annotations=READ_ONLY)
    def list_markets(limit: int = 20) -> list[dict[str, Any]]:
        """List the highest-volume active binary Polymarket markets."""
        return [
            {
                "id": m.id,
                "question": m.question,
                "yes_price": m.yes_price,
                "volume": round(m.volume),
                "end_date": m.end_date.isoformat() if m.end_date else None,
            }
            for m in get_service().list_markets(limit=limit)
        ]

    @mcp.tool(annotations=READ_ONLY)
    def search_evidence(
        market_id: str, query: str | None = None, k: int = 8
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant news evidence for a market (hybrid vector + BM25 search)."""
        return [
            {
                "title": r.chunk.title,
                "text": r.chunk.text,
                "url": r.chunk.url,
                "source": r.chunk.source,
                "score": round(r.score, 4),
                "published_at": r.chunk.published_at.isoformat() if r.chunk.published_at else None,
            }
            for r in get_service().search_evidence(market_id, query=query, k=k)
        ]

    @mcp.tool(annotations=READ_ONLY)
    def research_market(market_id: str, as_of: str | None = None) -> dict[str, Any]:
        """Run the agent: retrieve -> weigh evidence -> estimate P(YES) -> compare to market price.

        ``as_of`` (ISO date) restricts evidence to before that date for back-testing.
        """
        when = datetime.fromisoformat(as_of) if as_of else None
        return get_service().research_market(market_id, as_of=when).model_dump(mode="json")

    return mcp


def main(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765) -> None:
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def svc() -> ResearchService:
        return ResearchService()

    server = build_mcp(svc)
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
