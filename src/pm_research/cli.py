"""``pmra`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def _cmd_markets(args: argparse.Namespace) -> None:
    from .service import ResearchService

    for m in ResearchService().list_markets(limit=args.limit):
        price = f"{m.yes_price:.3f}" if m.yes_price is not None else "  n/a"
        print(f"{m.id:>8}  {price}  {m.question}")


def _cmd_research(args: argparse.Namespace) -> None:
    from .service import ResearchService

    as_of = datetime.fromisoformat(args.as_of) if args.as_of else None
    fc = ResearchService().research_market(args.market_id, as_of=as_of)
    print(fc.model_dump_json(indent=2))


def _cmd_eval_retrieval(args: argparse.Namespace) -> None:
    from .evals.retrieval_eval import run_retrieval_eval

    res = run_retrieval_eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if args.min_hybrid_recall5 is not None:
        got = res["modes"]["hybrid"]["recall@5"]
        if got < args.min_hybrid_recall5:
            sys.exit(f"hybrid recall@5 {got} < threshold {args.min_hybrid_recall5}")


def _cmd_eval_snapshot(args: argparse.Namespace) -> None:
    from .evals.forecast_eval import build_snapshot
    from .service import ResearchService

    snap = build_snapshot(
        ResearchService(),
        n_markets=args.n,
        horizon_days=args.horizon,
        end_date_min=args.end_date_min,
        end_date_max=args.end_date_max,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(snap, indent=1))
    print(f"wrote {len(snap['items'])} markets to {args.out}")


def _cmd_eval_forecast(args: argparse.Namespace) -> None:
    from .evals.forecast_eval import run_forecast_eval, to_markdown
    from .service import ResearchService

    snap = json.loads(Path(args.snapshot).read_text())
    # Offline replay: evidence comes from the snapshot, so no live sources are needed.
    report = run_forecast_eval(ResearchService(sources=[]), snap, out_dir=args.out)
    print(to_markdown(report))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="pmra", description="Prediction-market research agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("markets", help="list active markets")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=_cmd_markets)

    s = sub.add_parser("research", help="research one market")
    s.add_argument("market_id")
    s.add_argument("--as-of", help="ISO date; only use evidence published before it")
    s.set_defaults(func=_cmd_research)

    s = sub.add_parser("serve", help="run the FastAPI service (MCP mounted at /mcp)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(
        func=lambda a: __import__("uvicorn").run("pm_research.api:app", host=a.host, port=a.port)
    )

    s = sub.add_parser("mcp", help="run the MCP server")
    s.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(
        func=lambda a: __import__("pm_research.mcp_server", fromlist=["main"]).main(
            a.transport, a.host, a.port
        )
    )

    ev = sub.add_parser("eval", help="evaluation harness").add_subparsers(
        dest="eval_cmd", required=True
    )
    s = ev.add_parser("retrieval", help="recall@k / MRR on the labelled set (offline)")
    s.add_argument("--out", default="reports")
    s.add_argument("--min-hybrid-recall5", type=float, default=None)
    s.set_defaults(func=_cmd_eval_retrieval)

    s = ev.add_parser("snapshot", help="pull resolved markets + pre-cutoff evidence (network)")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--horizon", type=int, default=7)
    s.add_argument("--end-date-min", default=None)
    s.add_argument("--end-date-max", default=None)
    s.add_argument("--out", default="data/eval/forecast_snapshot.json")
    s.set_defaults(func=_cmd_eval_snapshot)

    s = ev.add_parser("forecast", help="score forecasts on a snapshot (offline replay)")
    s.add_argument("--snapshot", default="data/eval/forecast_snapshot.json")
    s.add_argument("--out", default="reports")
    s.set_defaults(func=_cmd_eval_forecast)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
