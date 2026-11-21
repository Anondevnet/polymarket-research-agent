"""Embedding backends behind one tiny protocol.

* ``hashing``: deterministic feature-hashed word uni/bi-grams. Zero dependencies, offline,
  used in tests/CI. Lexical, not semantic - a floor, not a recommendation.
* ``sentence-transformers``: local semantic model (``pip install .[local]``).
* ``openai``: OpenAI (or any compatible) embeddings API.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from typing import ClassVar, Protocol

import numpy as np

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
            for f in feats:
                idx, sign = self._bucket(f)
                out[row, idx] += sign
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - needs model download
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 384)
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class OpenAIEmbedder:  # pragma: no cover - needs API key
    _DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model: str, api_key: str | None, base_url: str | None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self.dim = self._DIMS.get(model, 1536)
        self.name = model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    model = settings.embedding_model
    if model.startswith("sentence-transformers/"):
        model = "text-embedding-3-small"
    return OpenAIEmbedder(model, settings.openai_api_key, settings.openai_base_url)
