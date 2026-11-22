"""Domain models shared across ingestion, retrieval, the agent, and evals."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Market(BaseModel):
    """A binary (Yes/No) Polymarket market, normalised from the Gamma API."""

    id: str
    question: str
    slug: str = ""
    description: str = ""
    end_date: datetime | None = None
    closed_time: datetime | None = None
    closed: bool = False
    yes_price: float | None = Field(default=None, description="Current/last YES price in [0,1]")
    volume: float = 0.0
    yes_token_id: str | None = None
    outcome: Literal[0, 1] | None = Field(default=None, description="Resolved outcome (1 = YES)")


class Document(BaseModel):
    """A raw piece of evidence fetched from a source (news article, post, ...)."""

    id: str
    source: str
    title: str
    text: str
    url: str = ""
    published_at: datetime | None = None
    market_id: str | None = None


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    title: str = ""
    url: str = ""
    source: str = ""
    published_at: datetime | None = None
    market_id: str | None = None


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str = "dense"


class EvidenceSummary(BaseModel):
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    summary: str = ""


class Forecast(BaseModel):
    market_id: str
    question: str
    probability: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    market_price: float | None = None
    edge: float | None = Field(default=None, description="probability - market_price")
    evidence: EvidenceSummary = Field(default_factory=EvidenceSummary)
    citations: list[str] = Field(default_factory=list)
    as_of: datetime | None = None
    llm: str | None = None
    trace_id: str | None = None
    trace_url: str | None = None
