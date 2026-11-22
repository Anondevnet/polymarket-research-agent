"""Web dashboard: a small static single-page app plus a few JSON endpoints.

No build step: plain HTML/CSS/JS served by FastAPI, with Chart.js loaded from a CDN.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .evals.reports import load_retrieval, load_runs, reports_dir
from .service import ResearchService
from .tracing import tracing_backend

STATIC = Path(__file__).parent / "web" / "static"


def mount_dashboard(app: FastAPI, get_service: Callable[[], ResearchService]) -> None:
    router = APIRouter()
    Service = Annotated[ResearchService, Depends(get_service)]

    @router.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/dashboard")

    @router.get("/dashboard", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @router.get("/api/config", tags=["dashboard"])
    def config(svc: Service) -> dict[str, Any]:
        s = svc.settings
        return {
            "version": __version__,
            "llm_provider": s.llm_provider,
            "llm": svc.llm.name,
            "embedder": svc.embedder.name,
            "vector_store": s.vector_store,
            "hybrid": s.hybrid,
            "sources": [src.name for src in svc.sources],
            "tracing": tracing_backend(),
            "read_only": True,
        }

    @router.get("/api/evals", tags=["dashboard"])
    def evals() -> dict[str, Any]:
        return {"forecast_runs": load_runs(), "retrieval": load_retrieval()}

    @router.get("/api/evals/{run}/calibration.png", tags=["dashboard"])
    def calibration_png(run: str) -> FileResponse:
        root = reports_dir().resolve()
        path = (root / run / "calibration.png").resolve()
        if root not in path.parents or not path.exists():
            raise HTTPException(404, "no such report")
        return FileResponse(path, media_type="image/png")

    app.include_router(router)
    app.mount("/dashboard/static", StaticFiles(directory=STATIC), name="dashboard-static")
