"""Forecast eval against resolved Polymarket markets.

For each resolved binary market we pick a forecast time ``as_of = close - horizon`` and:

* fetch evidence published strictly before ``as_of`` (look-ahead guard at fetch + read time),
* run the agent to get an independent P(YES),
* look up the market's own YES price at ``as_of`` from the CLOB price history,
* score agent, market, and a 50% baseline with Brier, log loss, ECE, plus a calibration plot.

``build_snapshot`` captures all network inputs to JSON so ``run_forecast_eval`` can be
replayed fully offline (that is what CI does).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..llm import FALLBACK_RATIONALE
from ..models import Document, Market
from ..service import ResearchService
from ..tracing import span
from .metrics import brier_score, expected_calibration_error, log_loss
from .plot import calibration_plot

log = logging.getLogger(__name__)


def forecast_time(market: Market, horizon_days: int) -> datetime | None:
    candidates = [d for d in (market.end_date, market.closed_time) if d is not None]
    if not candidates:
        return None
    return min(candidates) - timedelta(days=horizon_days)


def build_snapshot(
    svc: ResearchService,
    n_markets: int = 20,
    horizon_days: int = 7,
    end_date_min: str | None = None,
    end_date_max: str | None = None,
    min_volume: float = 250_000,
) -> dict[str, Any]:
    """Pull resolved markets + pre-cutoff evidence + market price at cutoff (network)."""
    markets = svc.polymarket.resolved_markets(
        limit=n_markets * 2,
        end_date_min=end_date_min,
        end_date_max=end_date_max,
        min_volume=min_volume,
    )
    items: list[dict[str, Any]] = []
    for m in markets:
        as_of = forecast_time(m, horizon_days)
        if as_of is None or m.yes_token_id is None:
            continue
        try:
            price = svc.polymarket.price_at(m.yes_token_id, as_of)
        except Exception as exc:
            log.warning("price history failed for %s: %s", m.id, exc)
            price = None
        if price is None:
            continue
        docs = svc.fetch_documents(m, as_of=as_of)
        items.append(
            {
                "market": m.model_dump(mode="json"),
                "as_of": as_of.isoformat(),
                "market_price": price,
                "documents": [d.model_dump(mode="json") for d in docs],
            }
        )
        log.info("snapshot %s: %d docs, price %.3f", m.id, len(docs), price)
        if len(items) >= n_markets:
            break
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "horizon_days": horizon_days,
        "sources": [s.name for s in svc.sources],
        "items": items,
    }


def _score(probs: list[float], outcomes: list[int]) -> dict[str, float]:
    return {
        "brier": round(brier_score(probs, outcomes), 4),
        "log_loss": round(log_loss(probs, outcomes), 4),
        "ece": round(expected_calibration_error(probs, outcomes, n_bins=5), 4),
    }


def run_forecast_eval(
    svc: ResearchService,
    snapshot: dict[str, Any],
    out_dir: str | Path = "reports",
    knowledge_cutoff: str | None = None,
    partial: bool = False,
) -> dict[str, Any]:
    """Score agent vs market vs 0.5 on a snapshot.

    ``knowledge_cutoff`` (ISO date): also report scores on the subset of markets whose
    forecast date is after the LLM's training cutoff, where the model cannot already know
    the outcome from pre-training.
    ``partial``: score only markets already in the checkpoint, making no new LLM calls
    (for reporting progress on a run that is still waiting on a rate limit).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint: each finished market is appended to rows.jsonl, and re-runs with the same
    # LLM skip it. Lets a rate-limited free-tier run resume instead of starting over.
    ckpt = out_dir / "rows.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("llm") == svc.llm.name:
                    done[r["market_id"]] = r
    rows: list[dict[str, Any]] = []
    with span("forecast_eval", as_type="evaluator", n=len(snapshot["items"]), llm=svc.llm.name):
        for item in snapshot["items"]:
            market = Market.model_validate(item["market"])
            if market.id in done:
                rows.append(done[market.id])
                continue
            if partial:
                continue
            as_of = datetime.fromisoformat(item["as_of"])
            docs = [Document.model_validate(d) for d in item["documents"]]
            svc.ingest_documents(market.id, docs)
            fc = svc.research(market, as_of=as_of, market_price=item["market_price"])
            rows.append(
                {
                    "market_id": market.id,
                    "question": market.question,
                    "as_of": item["as_of"],
                    "n_docs": len(docs),
                    "agent": fc.probability,
                    "market": item["market_price"],
                    "outcome": market.outcome,
                    "rationale": fc.rationale,
                    "llm": svc.llm.name,
                }
            )
            with ckpt.open("a") as fh:
                fh.write(json.dumps(rows[-1]) + "\n")
            log.info("scored %s (%d/%d)", market.id, len(rows), len(snapshot["items"]))
    outcomes = [int(r["outcome"]) for r in rows]
    forecasters = {
        f"agent ({svc.llm.name})": [r["agent"] for r in rows],
        "market price @ as_of": [r["market"] for r in rows],
        "baseline 0.5": [0.5] * len(rows),
    }

    def score_all(subset: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        ys = [int(r["outcome"]) for r in subset]
        return {
            f"agent ({svc.llm.name})": _score([r["agent"] for r in subset], ys),
            "market price @ as_of": _score([r["market"] for r in subset], ys),
            "baseline 0.5": _score([0.5] * len(subset), ys),
        }

    report: dict[str, Any] = {
        "n_markets": len(rows),
        "n_markets_in_snapshot": len(snapshot["items"]),
        "partial": len(rows) < len(snapshot["items"]),
        "horizon_days": snapshot.get("horizon_days"),
        "snapshot_created_at": snapshot.get("created_at"),
        "llm": svc.llm.name,
        "embedder": svc.embedder.name,
        "base_rate_yes": round(sum(outcomes) / len(outcomes), 4) if outcomes else None,
        "scores": score_all(rows) if rows else {},
        "n_llm_fallbacks": sum(r["rationale"] == FALLBACK_RATIONALE for r in rows),
        "rows": rows,
    }
    if knowledge_cutoff and rows:
        after = [r for r in rows if r["as_of"][:10] > knowledge_cutoff]
        report["knowledge_cutoff"] = knowledge_cutoff
        report["post_cutoff"] = {
            "n_markets": len(after),
            "base_rate_yes": round(sum(r["outcome"] for r in after) / len(after), 4)
            if after
            else None,
            "scores": score_all(after) if after else {},
        }
    if rows:
        calibration_plot(forecasters, outcomes, out_dir / "calibration.png")
    (out_dir / "forecast_eval.json").write_text(json.dumps(report, indent=2))
    (out_dir / "forecast_eval.md").write_text(to_markdown(report))
    return report


def to_markdown(report: dict[str, Any]) -> str:
    partial_note = (
        f" (PARTIAL: {report['n_markets']}/{report['n_markets_in_snapshot']} scored)"
        if report.get("partial")
        else ""
    )
    lines = [
        f"**Markets:** {report['n_markets']}{partial_note}"
        f" · **horizon:** {report['horizon_days']}d before close"
        f" · **LLM:** `{report['llm']}` · **embedder:** `{report['embedder']}`"
        f" · **base rate YES:** {report['base_rate_yes']}",
        "",
        "| Forecaster | Brier ↓ | Log loss ↓ | ECE ↓ |",
        "|---|---|---|---|",
    ]
    for name, s in report["scores"].items():
        lines.append(f"| {name} | {s['brier']} | {s['log_loss']} | {s['ece']} |")
    post = report.get("post_cutoff")
    if post and post["scores"]:
        lines += [
            "",
            f"**Subset forecast after LLM knowledge cutoff ({report['knowledge_cutoff']}):** "
            f"{post['n_markets']} markets · base rate YES: {post['base_rate_yes']}",
            "",
            "| Forecaster | Brier ↓ | Log loss ↓ | ECE ↓ |",
            "|---|---|---|---|",
        ]
        for name, s in post["scores"].items():
            lines.append(f"| {name} | {s['brier']} | {s['log_loss']} | {s['ece']} |")
    return "\n".join(lines) + "\n"
