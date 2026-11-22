"""Capture dashboard screenshots with headless Chromium (Playwright).

pip install playwright && playwright install chromium  # or: --chrome /usr/bin/google-chrome
python scripts/screenshots.py --base http://localhost:8000 --out dashboard-shots
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


def pick_market(base: str) -> str:
    with urllib.request.urlopen(f"{base}/markets?limit=60", timeout=60) as resp:
        markets = json.load(resp)
    contested = [m for m in markets if m.get("yes_price") and 0.15 <= m["yes_price"] <= 0.85]
    return str((contested or markets)[0]["id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out", default="dashboard-shots")
    ap.add_argument("--chrome", default=None, help="path to a system Chrome/Chromium binary")
    ap.add_argument("--market", default=None)
    ap.add_argument(
        "--no-research", action="store_true", help="skip the (LLM-calling) research run"
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    market = args.market or pick_market(args.base)

    with sync_playwright() as p:
        browser = (
            p.chromium.launch(executable_path=args.chrome) if args.chrome else p.chromium.launch()
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)

        page.goto(f"{args.base}/dashboard#/")
        page.wait_for_selector("#markets .ranked", timeout=60_000)
        page.screenshot(path=out / "01-markets.png")

        page.goto(f"{args.base}/dashboard#/market/{market}")
        page.wait_for_selector(
            "#evidence .ranked, #evidence .empty, #evidence .error", timeout=120_000
        )
        page.screenshot(path=out / "02-market-evidence.png", full_page=True)

        if not args.no_research:
            page.click("#run")
            page.wait_for_selector("#result .verdict, #result .error", timeout=300_000)
            page.screenshot(path=out / "03-research-result.png", full_page=True)
            page.goto(f"{args.base}/dashboard#/research")
            page.wait_for_selector("#pick option[value]", state="attached", timeout=60_000)
            page.screenshot(path=out / "05-research-hub.png", full_page=True)

        page.goto(f"{args.base}/dashboard#/evals")
        page.wait_for_selector("canvas", timeout=60_000)
        page.wait_for_timeout(800)  # let Chart.js finish its animation
        page.screenshot(path=out / "04-evals.png", full_page=True)
        browser.close()
    print(f"market {market}: wrote", sorted(str(f) for f in out.glob("*.png")))


if __name__ == "__main__":
    main()
