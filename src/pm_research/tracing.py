"""Tracing that is OpenTelemetry-native and Langfuse-aware, and a no-op when unconfigured.

* If ``LANGFUSE_PUBLIC_KEY``/``LANGFUSE_SECRET_KEY`` are set (and ``langfuse`` is installed),
  spans are recorded as Langfuse observations (Langfuse v3+ is itself built on OTel).
* Otherwise spans go through the global OpenTelemetry tracer. That is a no-op unless an
  exporter is configured (``OTEL_EXPORTER_OTLP_ENDPOINT`` or ``PMRA_OTEL_CONSOLE=true``).
* LangSmith tracing of the LangGraph run is enabled by LangChain itself via
  ``LANGSMITH_TRACING=true`` + ``LANGSMITH_API_KEY``; nothing to do here.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

log = logging.getLogger(__name__)

_langfuse: Any = None
_initialised = False

ObservationType = str  # "span" | "retriever" | "generation" | "agent" | "tool" | "evaluator"


def _safe(value: Any, limit: int = 4000) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        text = repr(value)
    return text[:limit]


def init_tracing(console: bool | None = None) -> None:
    """Idempotently configure exporters based on environment variables."""
    global _langfuse, _initialised
    if _initialised:
        return
    _initialised = True

    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import get_client

            _langfuse = get_client()
            log.info("Langfuse tracing enabled")
        except Exception as exc:  # pragma: no cover - optional dependency
            log.warning("Langfuse keys set but client unavailable: %s", exc)

    if console is None:
        console = os.getenv("PMRA_OTEL_CONSOLE", "").lower() in {"1", "true", "yes"}
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if _langfuse is None and (console or endpoint):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider = TracerProvider(resource=Resource.create({"service.name": "pm-research-agent"}))
        if console:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            except ImportError:  # pragma: no cover
                log.warning("Install the [tracing] extra for OTLP export")
        trace.set_tracer_provider(provider)


def langfuse_enabled() -> bool:
    return _langfuse is not None


class SpanHandle:
    """Small facade so call sites don't care which backend is active."""

    def __init__(self, otel_span: Any = None, lf_obs: Any = None) -> None:
        self._otel = otel_span
        self._lf = lf_obs

    def set(self, key: str, value: Any) -> None:
        if self._otel is not None:
            v = value if isinstance(value, (str, int, float, bool)) else _safe(value)
            self._otel.set_attribute(f"pmra.{key}", v)
        if self._lf is not None:
            self._lf.update(metadata={key: value})

    def output(self, value: Any) -> None:
        if self._otel is not None:
            self._otel.set_attribute("pmra.output", _safe(value))
        if self._lf is not None:
            self._lf.update(output=value)


@contextmanager
def span(
    name: str, as_type: ObservationType = "span", input: Any = None, **attrs: Any
) -> Iterator[SpanHandle]:
    """Open a traced span. Works with or without any tracing backend configured."""
    init_tracing()
    if _langfuse is not None:
        with _langfuse.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=attrs or None
        ) as obs:
            yield SpanHandle(lf_obs=obs)
        return
    tracer = trace.get_tracer("pm_research")
    with tracer.start_as_current_span(name) as otel_span:
        otel_span.set_attribute("pmra.observation_type", as_type)
        if input is not None:
            otel_span.set_attribute("pmra.input", _safe(input))
        handle = SpanHandle(otel_span=otel_span)
        for k, v in attrs.items():
            handle.set(k, v)
        yield handle


def current_trace_id() -> str | None:
    """Trace id of the active span (Langfuse or OTel), or None when tracing is a no-op."""
    if _langfuse is not None:
        try:
            tid = _langfuse.get_current_trace_id()
            return str(tid) if tid else None
        except Exception:  # pragma: no cover - defensive
            return None
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def trace_url(trace_id: str | None) -> str | None:
    """Link to a trace UI: Langfuse when enabled, else ``PMRA_TRACE_URL_TEMPLATE``
    (e.g. ``http://localhost:16686/trace/{trace_id}`` for Jaeger)."""
    if not trace_id:
        return None
    if _langfuse is not None:
        try:
            url = _langfuse.get_trace_url(trace_id=trace_id)
            return str(url) if url else None
        except Exception:  # pragma: no cover - defensive
            return None
    template = os.getenv("PMRA_TRACE_URL_TEMPLATE")
    return template.format(trace_id=trace_id) if template else None


def tracing_backend() -> str:
    """Human-readable name of the active tracing backend(s), for the dashboard."""
    init_tracing()
    parts = []
    if _langfuse is not None:
        parts.append("langfuse")
    if os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true"} and os.getenv(
        "LANGSMITH_API_KEY"
    ):
        parts.append("langsmith")
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("PMRA_OTEL_CONSOLE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        parts.append("otel")
    return ",".join(parts) or "none"


def flush() -> None:
    if _langfuse is not None:
        _langfuse.flush()
