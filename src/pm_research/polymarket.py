"""Read-only client for Polymarket's public Gamma (metadata) and CLOB (price history) APIs.

No authentication, no order placement: this project only *reads* market data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx

from .models import Market


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    if text.endswith("+00"):
        text += ":00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            out = json.loads(value)
            return out if isinstance(out, list) else []
        except json.JSONDecodeError:
            return []
    return []


def parse_market(raw: dict[str, Any]) -> Market | None:
    """Normalise a Gamma market payload. Returns None for non-binary markets."""
    outcomes = [str(o).lower() for o in _json_list(raw.get("outcomes"))]
    if outcomes != ["yes", "no"]:
        return None
    prices = [float(p) for p in _json_list(raw.get("outcomePrices"))]
    tokens = [str(t) for t in _json_list(raw.get("clobTokenIds"))]
    yes_price: float | None = prices[0] if prices else None
    closed = bool(raw.get("closed"))
    outcome: int | None = None
    resolved = raw.get("umaResolutionStatus") == "resolved" or closed
    if resolved and yes_price is not None:
        if yes_price >= 0.99:
            outcome = 1
        elif yes_price <= 0.01:
            outcome = 0
    return Market(
        id=str(raw["id"]),
        question=str(raw.get("question", "")),
        slug=str(raw.get("slug", "")),
        description=str(raw.get("description", "") or ""),
        end_date=_parse_dt(raw.get("endDate")),
        closed_time=_parse_dt(raw.get("closedTime")),
        closed=closed,
        yes_price=yes_price,
        volume=float(raw.get("volumeNum") or raw.get("volume") or 0.0),
        yes_token_id=tokens[0] if tokens else None,
        outcome=outcome,  # type: ignore[arg-type]
    )


class PolymarketClient:
    def __init__(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        timeout: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self._http = client or httpx.Client(timeout=timeout, headers={"User-Agent": "pmra/0.1"})

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._http.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def list_markets(
        self, limit: int = 20, active: bool = True, min_volume: float = 0.0
    ) -> list[Market]:
        params: dict[str, Any] = {
            "limit": limit,
            "closed": "false" if active else "true",
            "order": "volumeNum",
            "ascending": "false",
        }
        if active:
            params["active"] = "true"
        if min_volume:
            params["volume_num_min"] = min_volume
        raw = self._get(f"{self.gamma_url}/markets", params)
        return [m for m in (parse_market(r) for r in raw) if m is not None]

    def get_market(self, market_id: str) -> Market:
        raw = self._get(f"{self.gamma_url}/markets/{market_id}")
        market = parse_market(raw)
        if market is None:
            raise ValueError(f"Market {market_id} is not a binary Yes/No market")
        return market

    def resolved_markets(
        self,
        limit: int = 50,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        min_volume: float = 100_000.0,
    ) -> list[Market]:
        """Closed binary markets with a clean 0/1 resolution, highest volume first."""
        params: dict[str, Any] = {
            "closed": "true",
            "limit": min(limit * 3, 500),
            "order": "volumeNum",
            "ascending": "false",
            "volume_num_min": min_volume,
        }
        if end_date_min:
            params["end_date_min"] = end_date_min
        if end_date_max:
            params["end_date_max"] = end_date_max
        raw = self._get(f"{self.gamma_url}/markets", params)
        out = [m for m in (parse_market(r) for r in raw) if m and m.outcome is not None]
        return out[:limit]

    def price_at(self, token_id: str, when: datetime, window_hours: int = 48) -> float | None:
        """Last traded YES price at or before ``when`` (from CLOB price history)."""
        end_ts = int(when.timestamp())
        params = {
            "market": token_id,
            "startTs": end_ts - window_hours * 3600,
            "endTs": end_ts,
            "fidelity": 60,
        }
        data = self._get(f"{self.clob_url}/prices-history", params)
        history = [h for h in data.get("history", []) if h.get("t", 0) <= end_ts]
        if not history:
            return None
        return float(history[-1]["p"])
