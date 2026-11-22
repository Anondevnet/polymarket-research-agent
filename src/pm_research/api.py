"""FastAPI service exposing read-only research endpoints (and the MCP server at /mcp)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import datetime
from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import __version__
from .models import Forecast, Market, ScoredChunk
from .service import ResearchService


@lru_cache(maxsize=1)
def get_service() -> ResearchService:
    return ResearchService()


Service = Annotated[ResearchService, Depends(get_service)]


class ResearchRequest(BaseModel):
    as_of: datetime | None = None


def create_app(mount_mcp: bool = True) -> FastAPI:
    mcp_app = None
    if mount_mcp:
        from .mcp_server import build_mcp

        mcp = build_mcp(get_service)
        mcp_app = mcp.streamable_http_app(
            streamable_http_path="/", stateless_http=True, host="0.0.0.0"
        )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if mcp_app is not None:
            async with mcp_app.router.lifespan_context(mcp_app):
                yield
        else:
            yield

    app = FastAPI(
        title="Prediction-Market Research Agent",
        version=__version__,
        description="Read-only research over Polymarket markets. No trading.",
        lifespan=lifespan,
    )
    if mcp_app is not None:
        app.mount("/mcp", mcp_app)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/markets", response_model=list[Market])
    def markets(svc: Service, limit: int = Query(20, ge=1, le=100)) -> list[Market]:
        return svc.list_markets(limit=limit)

    @app.get("/markets/{market_id}", response_model=Market)
    def market(svc: Service, market_id: str) -> Market:
        try:
            return svc.get_market(market_id)
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/markets/{market_id}/evidence", response_model=list[ScoredChunk])
    def evidence(
        svc: Service, market_id: str, q: str | None = None, k: int = Query(8, ge=1, le=50)
    ) -> list[ScoredChunk]:
        try:
            return svc.search_evidence(market_id, query=q, k=k)
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/markets/{market_id}/research", response_model=Forecast)
    def research(svc: Service, market_id: str, body: ResearchRequest | None = None) -> Forecast:
        try:
            return svc.research_market(market_id, as_of=body.as_of if body else None)
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc

    from .dashboard import mount_dashboard

    mount_dashboard(app, get_service)
    return app


app = create_app()
