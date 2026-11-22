import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pm_research import api
from pm_research.evals.reports import load_runs
from pm_research.service import ResearchService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(service: ResearchService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PMRA_REPORTS_DIR", str(ROOT / "reports"))
    app = api.create_app(mount_mcp=False)
    app.dependency_overrides[api.get_service] = lambda: service
    return TestClient(app)


def test_root_redirects_to_dashboard(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/dashboard"


def test_dashboard_page_and_assets(client: TestClient) -> None:
    page = client.get("/dashboard")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/dashboard/static/app.js" in page.text
    for asset in ("app.js", "styles.css"):
        assert client.get(f"/dashboard/static/{asset}").status_code == 200


def test_config_reports_offline_defaults(client: TestClient) -> None:
    c = client.get("/api/config").json()
    assert c["llm_provider"] == "heuristic" and c["llm"] == "heuristic-lexicon"
    assert c["read_only"] is True and c["tracing"] == "none"
    assert "key" not in json.dumps(c).lower()  # never leak credentials


def test_market_detail_endpoint(client: TestClient) -> None:
    assert client.get("/markets/m1").json()["question"].startswith("Will the Federal Reserve")
    assert client.get("/markets/nope").status_code == 404


def test_research_includes_llm_and_trace_fields(client: TestClient) -> None:
    f = client.post("/markets/m1/research", json={}).json()
    assert f["llm"] == "heuristic-lexicon"
    assert f["trace_url"] is None  # no tracing backend configured offline


def test_evals_api_serves_committed_reports(client: TestClient) -> None:
    data = client.get("/api/evals").json()
    runs = {r["name"]: r for r in data["forecast_runs"]}
    assert {"heuristic", "gemini-2.5-flash-lite"} <= set(runs)
    committed = json.loads((ROOT / "reports/heuristic/forecast_eval.json").read_text())
    assert runs["heuristic"]["scores"] == committed["scores"]  # numbers are passed through
    g = runs["gemini-2.5-flash-lite"]
    assert g["partial"] is True and g["n_markets"] < g["n_markets_in_snapshot"]
    assert any("same" in k for k in g["same_sample_comparisons"])
    assert data["retrieval"]["modes"]["hybrid"]["recall@5"] > 0
    png = client.get("/api/evals/heuristic/calibration.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"


def test_calibration_png_rejects_traversal(client: TestClient) -> None:
    assert client.get("/api/evals/..%2F..%2Fpyproject.toml/calibration.png").status_code == 404
    assert client.get("/api/evals/missing/calibration.png").status_code == 404


def test_load_runs_same_sample_matches_metrics(tmp_path: Path) -> None:
    for name, agent, partial in (("full", [0.2, 0.8, 0.5], False), ("part", [0.9], True)):
        d = tmp_path / name
        d.mkdir()
        rows = [
            {
                "market_id": str(i),
                "question": "q",
                "as_of": "2025-01-01",
                "agent": a,
                "market": 0.5,
                "outcome": 1,
                "rationale": "",
                "llm": name,
            }
            for i, a in enumerate(agent)
        ]
        (d / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (d / "forecast_eval.json").write_text(
            json.dumps(
                {
                    "llm": name,
                    "n_markets": len(rows),
                    "n_markets_in_snapshot": 3,
                    "partial": partial,
                    "scores": {},
                }
            )
        )
    runs = {r["name"]: r for r in load_runs(tmp_path)}
    comp = runs["part"]["same_sample_comparisons"]
    assert comp == {"agent (full), same 1 markets": {"brier": 0.64, "log_loss": 1.6094, "ece": 0.8}}
