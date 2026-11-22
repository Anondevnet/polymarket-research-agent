"""Read committed eval reports (``reports/*/forecast_eval.json``) for the dashboard.

Only numbers that exist in the report files are returned; derived numbers (calibration bins,
same-sample comparisons) are recomputed from the per-market rows with the same metrics code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .metrics import brier_score, calibration_bins, expected_calibration_error, log_loss


def reports_dir() -> Path:
    """$PMRA_REPORTS_DIR, else ./reports, else the repo checkout's reports/."""
    if env := os.getenv("PMRA_REPORTS_DIR"):
        return Path(env)
    cwd = Path.cwd() / "reports"
    return cwd if cwd.exists() else Path(__file__).resolve().parents[3] / "reports"


def _rows(run_dir: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    ckpt = run_dir / "rows.jsonl"
    if ckpt.exists():
        return [json.loads(x) for x in ckpt.read_text().splitlines() if x.strip()]
    rows: list[dict[str, Any]] = report.get("rows", [])
    return rows


def _scores(probs: list[float], ys: list[int]) -> dict[str, float]:
    return {
        "brier": round(brier_score(probs, ys), 4),
        "log_loss": round(log_loss(probs, ys), 4),
        "ece": round(expected_calibration_error(probs, ys, n_bins=5), 4),
    }


def load_runs(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or reports_dir()
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    rows_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(root.glob("*/forecast_eval.json")):
        report = json.loads(path.read_text())
        rows = _rows(path.parent, report)
        name = path.parent.name
        rows_by_run[name] = {r["market_id"]: r for r in rows}
        ys = [int(r["outcome"]) for r in rows]
        series = {
            f"agent ({report.get('llm')})": [float(r["agent"]) for r in rows],
            "market price @ as_of": [float(r["market"]) for r in rows],
        }
        runs.append(
            {
                "name": name,
                "llm": report.get("llm"),
                "embedder": report.get("embedder"),
                "n_markets": report.get("n_markets"),
                "n_markets_in_snapshot": report.get(
                    "n_markets_in_snapshot", report.get("n_markets")
                ),
                "partial": bool(report.get("partial", False)),
                "horizon_days": report.get("horizon_days"),
                "base_rate_yes": report.get("base_rate_yes"),
                "snapshot_created_at": report.get("snapshot_created_at"),
                "knowledge_cutoff": report.get("knowledge_cutoff"),
                "scores": report.get("scores", {}),
                "post_cutoff": report.get("post_cutoff"),
                "n_llm_fallbacks": report.get("n_llm_fallbacks"),
                "calibration": {
                    k: calibration_bins(v, ys, n_bins=5) for k, v in series.items() if rows
                },
                "has_png": (path.parent / "calibration.png").exists(),
                "rows": [
                    {
                        k: r.get(k)
                        for k in (
                            "market_id",
                            "question",
                            "as_of",
                            "agent",
                            "market",
                            "outcome",
                            "rationale",
                        )
                    }
                    for r in rows
                ],
            }
        )
    # Same-sample comparison: score other runs' agents on each partial run's markets.
    for run in runs:
        if not run["partial"]:
            continue
        ids = [r["market_id"] for r in run["rows"]]
        ys = [int(r["outcome"]) for r in run["rows"]]
        comparisons = {}
        for other, by_id in rows_by_run.items():
            if other == run["name"] or not all(i in by_id for i in ids):
                continue
            comparisons[f"agent ({other}), same {len(ids)} markets"] = _scores(
                [float(by_id[i]["agent"]) for i in ids], ys
            )
        run["same_sample_comparisons"] = comparisons
    return runs


def load_retrieval(root: Path | None = None) -> dict[str, Any] | None:
    path = (root or reports_dir()) / "retrieval_eval.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data
