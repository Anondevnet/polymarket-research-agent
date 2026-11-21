import json
from pathlib import Path

from conftest import DOCS, MARKET, dt
from pm_research.evals.forecast_eval import forecast_time, run_forecast_eval
from pm_research.evals.retrieval_eval import run_retrieval_eval
from pm_research.service import ResearchService

ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_eval_runs_offline() -> None:
    res = run_retrieval_eval()
    assert set(res["modes"]) == {"bm25", "dense", "hybrid"}
    assert res["modes"]["hybrid"]["recall@5"] >= 0.8  # regression floor for CI


def test_forecast_time_uses_earliest_close() -> None:
    m = MARKET.model_copy(update={"closed_time": dt("2025-09-01T00:00:00")})
    assert forecast_time(m, 7) == dt("2025-08-25T00:00:00")


def test_forecast_eval_writes_report(service: ResearchService, tmp_path: Path) -> None:
    snap = {
        "horizon_days": 7,
        "created_at": "x",
        "items": [
            {
                "market": MARKET.model_copy(update={"id": mid, "outcome": out}).model_dump(
                    mode="json"
                ),
                "as_of": "2025-09-10T00:00:00+00:00",
                "market_price": price,
                "documents": [d.model_dump(mode="json") for d in DOCS],
            }
            for mid, out, price in [("a", 1, 0.7), ("b", 0, 0.2), ("c", 1, 0.55)]
        ],
    }
    report = run_forecast_eval(service, snap, out_dir=tmp_path)
    assert report["n_markets"] == 3
    assert report["scores"]["baseline 0.5"]["brier"] == 0.25
    assert (tmp_path / "calibration.png").stat().st_size > 1000
    assert json.loads((tmp_path / "forecast_eval.json").read_text())["n_markets"] == 3


def test_committed_snapshot_is_valid() -> None:
    snap = json.loads((ROOT / "data/eval/forecast_snapshot.json").read_text())
    assert snap["items"], "snapshot should contain markets"
    for it in snap["items"]:
        assert it["market"]["outcome"] in (0, 1)
        for d in it["documents"]:  # look-ahead guard held when the snapshot was built
            assert d["published_at"] < it["as_of"]
