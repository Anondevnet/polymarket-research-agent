from fastapi.testclient import TestClient

from pm_research import api
from pm_research.mcp_server import build_mcp
from pm_research.service import ResearchService


def _client(service: ResearchService) -> TestClient:
    app = api.create_app(mount_mcp=False)
    app.dependency_overrides[api.get_service] = lambda: service
    return TestClient(app)


def test_health(service: ResearchService) -> None:
    assert _client(service).get("/healthz").json()["status"] == "ok"


def test_markets_and_research(service: ResearchService) -> None:
    c = _client(service)
    assert c.get("/markets").json()[0]["id"] == "m1"
    ev = c.get("/markets/m1/evidence", params={"q": "Powell", "k": 2}).json()
    assert len(ev) == 2
    fc = c.post("/markets/m1/research", json={"as_of": "2025-09-10T00:00:00Z"}).json()
    assert 0 < fc["probability"] < 1 and fc["market_price"] == 0.62
    assert c.post("/markets/nope/research").status_code == 404


def test_app_mounts_mcp() -> None:
    app = api.create_app(mount_mcp=True)
    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)


async def test_mcp_tools(service: ResearchService) -> None:
    mcp = build_mcp(lambda: service)
    names = {t.name for t in await mcp.list_tools()}
    assert names == {"list_markets", "search_evidence", "research_market"}
    res = await mcp.call_tool(
        "research_market", {"market_id": "m1", "as_of": "2025-09-10T00:00:00+00:00"}
    )
    assert not getattr(res, "isError", getattr(res, "is_error", False))
