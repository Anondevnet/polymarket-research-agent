from typing import Any

import pytest

from conftest import MARKET, FakeOpenAI
from pm_research.config import Settings
from pm_research.llm import (
    FALLBACK_RATIONALE,
    GEMINI_DEFAULT_MODEL,
    GEMINI_OPENAI_BASE_URL,
    OpenAIChatLLM,
    build_llm,
)
from pm_research.models import EvidenceSummary


class RateLimitError(Exception):
    """Stand-in for openai.RateLimitError (matched by class name / status code)."""

    status_code = 429

    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("429 Too Many Requests")
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = type("Resp", (), {"status_code": 429, "headers": headers})()


class FlakyClient(FakeOpenAI):
    def __init__(self, failures: list[Exception], replies: list[str]) -> None:
        super().__init__(replies)
        inner = self.chat.completions
        failures = list(failures)

        class _Completions:
            def create(self, **kwargs: Any) -> Any:
                if failures:
                    raise failures.pop(0)
                return inner.create(**kwargs)

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_retries_on_rate_limit_with_backoff() -> None:
    sleeps: list[float] = []
    client = FlakyClient(
        [RateLimitError(), RateLimitError(retry_after="7")],
        ['{"probability": 0.3, "rationale": "r"}'],
    )
    llm = OpenAIChatLLM("m", client=client, sleep=sleeps.append, backoff_base_s=1.0)
    p, why = llm.estimate(MARKET, EvidenceSummary(), None)
    assert (p, why) == (0.3, "r")
    assert len(sleeps) == 2 and sleeps[1] >= 7.0  # honours Retry-After


def test_gives_up_after_max_retries() -> None:
    client = FlakyClient([RateLimitError()] * 3, ["{}"])
    llm = OpenAIChatLLM("m", client=client, sleep=lambda _: None, max_retries=2)
    with pytest.raises(RateLimitError):
        llm.estimate(MARKET, EvidenceSummary(), None)


def test_non_retryable_error_raises_immediately() -> None:
    client = FlakyClient([ValueError("bad request")], ["{}"])
    llm = OpenAIChatLLM("m", client=client, sleep=lambda _: None)
    with pytest.raises(ValueError):
        llm.estimate(MARKET, EvidenceSummary(), None)


def test_pacing_between_calls() -> None:
    sleeps: list[float] = []
    client = FakeOpenAI(['{"probability": 0.4}', '{"probability": 0.6}'])
    llm = OpenAIChatLLM("m", client=client, sleep=sleeps.append, min_interval_s=5.0)
    llm.estimate(MARKET, EvidenceSummary(), None)
    llm.estimate(MARKET, EvidenceSummary(), None)
    assert sleeps and 4.0 < sleeps[-1] <= 5.0


def test_missing_probability_is_flagged_as_fallback() -> None:
    llm = OpenAIChatLLM("m", client=FakeOpenAI(['{"rationale": "no number"}']))
    assert llm.estimate(MARKET, EvidenceSummary(), None) == (0.5, FALLBACK_RATIONALE)


def test_gemini_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    s = Settings(llm_provider="gemini", gemini_api_key="test-key", llm_min_interval_s=6.5)
    llm = build_llm(s)
    assert isinstance(llm, OpenAIChatLLM)
    assert llm.name == GEMINI_DEFAULT_MODEL  # default gpt model name swapped for Gemini
    assert str(llm._client.base_url) == GEMINI_OPENAI_BASE_URL
    assert llm.min_interval_s == 6.5
    s2 = Settings(llm_provider="gemini", gemini_api_key="k", llm_model="gemini-2.5-flash-lite")
    assert build_llm(s2).name == "gemini-2.5-flash-lite"


def test_daily_quota_is_not_retried() -> None:
    class DailyQuota(RateLimitError):
        def __str__(self) -> str:
            return "Generate content requests per day limit exceeded on Free Tier"

    sleeps: list[float] = []
    client = FlakyClient([DailyQuota()], ["{}"])
    llm = OpenAIChatLLM("m", client=client, sleep=sleeps.append)
    with pytest.raises(DailyQuota):
        llm.estimate(MARKET, EvidenceSummary(), None)
    assert sleeps == []
