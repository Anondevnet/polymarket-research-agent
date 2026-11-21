"""Runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PMRA_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Models
    embedding_provider: Literal["hashing", "sentence-transformers", "openai"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 512  # only used by the hashing embedder
    llm_provider: Literal["heuristic", "openai", "gemini"] = "heuristic"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_retries: int = 6  # retries on 429 / 5xx / connection errors (exp. backoff)
    llm_min_interval_s: float = 0.0  # pacing between calls, e.g. 6.5 for a 10 RPM free tier
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")

    # Vector store / retrieval
    vector_store: Literal["qdrant", "memory"] = "qdrant"
    qdrant_url: str = ":memory:"
    qdrant_api_key: str | None = None
    collection: str = "pm_evidence"
    hybrid: bool = True
    top_k: int = 8
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Sources
    sources: str = "google_news,gdelt"
    max_docs_per_source: int = 25
    http_timeout: float = 20.0

    # Polymarket (read-only public APIs)
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"

    # Tracing
    otel_console: bool = False

    @property
    def source_names(self) -> list[str]:
        return [s.strip() for s in self.sources.split(",") if s.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
