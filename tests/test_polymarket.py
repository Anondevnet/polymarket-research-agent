import json
from datetime import UTC, datetime

import httpx
import respx

from pm_research.polymarket import PolymarketClient, parse_market

RAW = {
    "id": "253591",
    "question": "Will Donald Trump win the 2024 US Presidential Election?",
    "slug": "trump-2024",
    "endDate": "2024-11-05T12:00:00Z",
    "closedTime": "2024-11-06 15:17:41+00",
    "outcomes": json.dumps(["Yes", "No"]),
    "outcomePrices": json.dumps(["1", "0"]),
    "clobTokenIds": json.dumps(["111", "222"]),
    "volumeNum": 1.5e9,
    "closed": True,
    "umaResolutionStatus": "resolved",
}


def test_parse_resolved_market() -> None:
    m = parse_market(RAW)
    assert m is not None
    assert m.outcome == 1 and m.yes_token_id == "111" and m.closed
    assert m.closed_time == datetime(2024, 11, 6, 15, 17, 41, tzinfo=UTC)


def test_parse_skips_non_binary() -> None:
    assert parse_market({**RAW, "outcomes": json.dumps(["A", "B", "C"])}) is None


def test_parse_open_market_has_no_outcome() -> None:
    m = parse_market(
        {
            **RAW,
            "closed": False,
            "umaResolutionStatus": None,
            "outcomePrices": json.dumps(["0.6", "0.4"]),
        }
    )
    assert m is not None and m.outcome is None and m.yes_price == 0.6


@respx.mock
def test_client_endpoints() -> None:
    respx.get("https://gamma-api.polymarket.com/markets").mock(
        return_value=httpx.Response(
            200, json=[RAW, {**RAW, "id": "2", "outcomePrices": '["0.5","0.5"]'}]
        )
    )
    respx.get("https://gamma-api.polymarket.com/markets/253591").mock(
        return_value=httpx.Response(200, json=RAW)
    )
    respx.get("https://clob.polymarket.com/prices-history").mock(
        return_value=httpx.Response(
            200, json={"history": [{"t": 100, "p": 0.4}, {"t": 200, "p": 0.65}]}
        )
    )
    c = PolymarketClient()
    resolved = c.resolved_markets(limit=5)
    assert [m.id for m in resolved] == ["253591"]  # ambiguous 0.5/0.5 market excluded
    assert c.get_market("253591").question.startswith("Will Donald")
    assert c.price_at("111", datetime.fromtimestamp(250, tz=UTC)) == 0.65
