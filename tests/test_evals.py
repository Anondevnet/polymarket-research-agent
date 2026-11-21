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
    snap["items"][0]["as_of"] = "2025-01-10T00:00:00+00:00"
    report = run_forecast_eval(service, snap, out_dir=tmp_path, knowledge_cutoff="2025-02-01")
    assert report["n_markets"] == 3
    assert report["post_cutoff"]["n_markets"] == 2
    assert report["n_llm_fallbacks"] == 0
    assert "after LLM knowledge cutoff" in (tmp_path / "forecast_eval.md").read_text()
    # checkpoint: a re-run reuses finished rows instead of calling the LLM again
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 3
    service.research = None  # type: ignore[method-assign,assignment]
    again = run_forecast_eval(service, snap, out_dir=tmp_path)
    assert again["scores"] == {k: v for k, v in report["scores"].items()}
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


def test_partial_scores_only_checkpointed(service: ResearchService, tmp_path: Path) -> None:
    items = [
        {
            "market": MARKET.model_copy(update={"id": mid, "outcome": 1}).model_dump(mode="json"),
            "as_of": "2025-09-10T00:00:00+00:00",
            "market_price": 0.6,
            "documents": [d.model_dump(mode="json") for d in DOCS],
        }
        for mid in ("a", "b")
    ]
    run_forecast_eval(service, {"items": items[:1]}, out_dir=tmp_path)
    service.research = None  # type: ignore[method-assign,assignment]
    rep = run_forecast_eval(service, {"items": items}, out_dir=tmp_path, partial=True)
    assert rep["n_markets"] == 1 and rep["partial"] is True
    assert "PARTIAL: 1/2" in (tmp_path / "forecast_eval.md").read_text()
