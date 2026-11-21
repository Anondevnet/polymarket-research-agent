"""LLM backends: an OpenAI-compatible chat model, and an offline heuristic baseline.

The heuristic exists so the whole pipeline (graph, API, MCP, evals) runs with no keys and
deterministically in CI. It is a lexicon baseline and is labelled as such in eval reports.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from datetime import datetime
from typing import Any, Protocol

from .config import Settings
from .models import EvidenceSummary, Market, ScoredChunk
from .prompts import ESTIMATE_SYSTEM, SUMMARIZE_SYSTEM, format_snippets
from .tracing import span

log = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_RATIONALE = "(fallback: unparseable LLM output, using 0.5)"

POSITIVE = {
    "wins", "win", "won", "leads", "lead", "leading", "ahead", "surges", "surge", "soars",
    "rally", "rallies", "approves", "approved", "approval", "passes", "passed", "confirmed",
    "confirms", "likely", "favored", "favorite", "gains", "record", "beats", "signs", "signed",
    "agrees", "agreement", "deal", "victory", "secures", "clinches", "rises", "jumps", "boost",
    "expected", "poised", "set", "announces", "launch", "launches", "breakthrough", "higher",
}  # fmt: skip
NEGATIVE = {
    "loses", "lose", "lost", "trails", "trailing", "behind", "falls", "fall", "drops", "plunges",
    "plunge", "slump", "rejects", "rejected", "fails", "failed", "denies", "denied", "unlikely",
    "delays", "delayed", "blocked", "blocks", "cancels", "cancelled", "canceled", "crash",
    "crashes", "withdraws", "suspends", "ruled", "doubt", "doubts", "collapse", "lower",
    "postponed", "stalls", "stalled", "no", "not", "won't", "never", "defeat", "defeated",
}  # fmt: skip


class LLM(Protocol):
    name: str

    def summarize(self, market: Market, chunks: list[ScoredChunk]) -> EvidenceSummary: ...

    def estimate(
        self, market: Market, evidence: EvidenceSummary, as_of: datetime | None
    ) -> tuple[float, str]: ...


def _snippets(chunks: list[ScoredChunk]) -> list[tuple[str, str, str]]:
    return [
        (
            c.chunk.title,
            c.chunk.published_at.date().isoformat() if c.chunk.published_at else "undated",
            c.chunk.text[:600],
        )
        for c in chunks
    ]


def _clip(p: float) -> float:
    return min(0.98, max(0.02, p))


class HeuristicLLM:
    """Lexicon-based stance scoring. Deterministic, offline, and intentionally simple."""

    name = "heuristic-lexicon"

    def __init__(self, prior: float = 0.5, strength: float = 0.35) -> None:
        self.prior = prior
        self.strength = strength

    @staticmethod
    def stance(text: str) -> int:
        words = re.findall(r"[a-z']+", text.lower())
        return sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)

    def summarize(self, market: Market, chunks: list[ScoredChunk]) -> EvidenceSummary:
        ev_for, ev_against = [], []
        for i, c in enumerate(chunks, 1):
            s = self.stance(c.chunk.text)
            line = f"{c.chunk.title or c.chunk.text[:120]} [{i}]"
            if s > 0:
                ev_for.append(line)
            elif s < 0:
                ev_against.append(line)
        summary = (
            f"{len(ev_for)} snippet(s) lean YES and {len(ev_against)} lean NO out of "
            f"{len(chunks)} retrieved (lexicon heuristic)."
        )
        return EvidenceSummary(evidence_for=ev_for, evidence_against=ev_against, summary=summary)

    def estimate(
        self, market: Market, evidence: EvidenceSummary, as_of: datetime | None
    ) -> tuple[float, str]:
        net = len(evidence.evidence_for) - len(evidence.evidence_against)
        logit = math.log(self.prior / (1 - self.prior)) + self.strength * net
        p = _clip(1 / (1 + math.exp(-logit)))
        return p, f"Heuristic: prior {self.prior:.2f}, net stance {net:+d} -> {p:.2f}."


def _parse_json(text: str) -> dict[str, Any]:
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                out = json.loads(m.group(0))
                return out if isinstance(out, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _retryable(exc: Exception) -> tuple[bool, float | None]:
    """Is this a transient API error? Returns (retryable, server-suggested delay)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__
    text = str(getattr(exc, "body", "") or exc)
    if "per day" in text.lower() or "PerDay" in text:
        return False, None  # daily quota: retrying within this run cannot succeed
    transient = status in {408, 409, 429, 500, 502, 503, 504} or name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
    delay: float | None = None
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        if "retry-after" in headers:
            delay = float(headers["retry-after"])
    except (TypeError, ValueError):
        delay = None
    return transient, delay


class OpenAIChatLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, Gemini, Ollama, vLLM, xAI, ...).

    Includes client-side pacing (``min_interval_s``) and exponential backoff with jitter on
    429/5xx so it behaves on free-tier rate limits.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        client: Any = None,
        max_retries: int = 6,
        min_interval_s: float = 0.0,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        sleep: Any = time.sleep,
    ) -> None:
        if client is None:
            from openai import OpenAI

            # Retries are handled here so pacing/backoff are explicit and traceable.
            client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=0)
        self._client = client
        self.model = model
        self.temperature = temperature
        self.name = model
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep
        self._last_call = 0.0

    def _create(self, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                transient, suggested = _retryable(exc)
                if not transient or attempt >= self.max_retries:
                    raise
                backoff = min(self.backoff_max_s, self.backoff_base_s * 2**attempt)
                delay = max(suggested or 0.0, backoff * (0.5 + random.random() / 2))
                attempt += 1
                log.warning(
                    "LLM call failed (%s); retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)

    def _chat(self, name: str, system: str, user: str) -> dict[str, Any]:
        with span(name, as_type="generation", input=user, model=self.model) as s:
            resp = self._create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            content = resp.choices[0].message.content or "{}"
            usage = getattr(resp, "usage", None)
            if usage is not None:
                s.set("prompt_tokens", getattr(usage, "prompt_tokens", 0))
                s.set("completion_tokens", getattr(usage, "completion_tokens", 0))
            s.output(content)
            return _parse_json(content)

    def summarize(self, market: Market, chunks: list[ScoredChunk]) -> EvidenceSummary:
        user = (
            f"Question: {market.question}\nResolution criteria: {market.description[:1500]}\n\n"
            f"Evidence:\n{format_snippets(_snippets(chunks)) or '(none found)'}"
        )
        data = self._chat("summarize_evidence", SUMMARIZE_SYSTEM, user)
        return EvidenceSummary(
            evidence_for=[str(x) for x in data.get("evidence_for", [])],
            evidence_against=[str(x) for x in data.get("evidence_against", [])],
            summary=str(data.get("summary", "")),
        )

    def estimate(
        self, market: Market, evidence: EvidenceSummary, as_of: datetime | None
    ) -> tuple[float, str]:
        user = (
            f"Forecast date: {as_of.date().isoformat() if as_of else 'today'}\n"
            f"Market closes: {market.end_date.date().isoformat() if market.end_date else 'n/a'}\n"
            f"Question: {market.question}\nResolution criteria: {market.description[:1500]}\n\n"
            f"Evidence for YES:\n- "
            + "\n- ".join(evidence.evidence_for or ["(none)"])
            + "\nEvidence against YES:\n- "
            + "\n- ".join(evidence.evidence_against or ["(none)"])
            + f"\nSummary: {evidence.summary}"
        )
        data = self._chat("estimate_probability", ESTIMATE_SYSTEM, user)
        try:
            p = float(data["probability"])
        except (KeyError, TypeError, ValueError):
            return 0.5, FALLBACK_RATIONALE
        return _clip(p), str(data.get("rationale", ""))


def build_llm(settings: Settings) -> LLM:
    common = {
        "temperature": settings.llm_temperature,
        "max_retries": settings.llm_max_retries,
        "min_interval_s": settings.llm_min_interval_s,
    }
    if settings.llm_provider == "gemini":
        # Gemini via its OpenAI-compatible endpoint; key from GEMINI_API_KEY.
        model = settings.llm_model
        if not model.startswith("gemini"):
            model = GEMINI_DEFAULT_MODEL
        return OpenAIChatLLM(
            model,
            settings.gemini_api_key or settings.openai_api_key,
            settings.openai_base_url or GEMINI_OPENAI_BASE_URL,
            **common,  # type: ignore[arg-type]
        )
    if settings.llm_provider == "openai":
        return OpenAIChatLLM(
            settings.llm_model,
            settings.openai_api_key,
            settings.openai_base_url,
            **common,  # type: ignore[arg-type]
        )
    return HeuristicLLM()
