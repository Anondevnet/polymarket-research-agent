"""Vector stores: Qdrant (server, on-disk, or ``:memory:``) and a numpy in-memory fallback."""

from __future__ import annotations

import uuid
from typing import Any, Protocol

import numpy as np

from .config import Settings
from .models import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self, vector: np.ndarray, k: int = 8, market_id: str | None = None
    ) -> list[ScoredChunk]: ...

    def all_chunks(self, market_id: str | None = None) -> list[Chunk]: ...

    def count(self, market_id: str | None = None) -> int: ...


class InMemoryVectorStore:
    """Brute-force cosine search. Fine for thousands of chunks; used when Qdrant is unwanted."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for c, v in zip(chunks, vectors, strict=True):
            self._chunks[c.id] = c
            n = np.linalg.norm(v)
            self._vecs[c.id] = v / n if n else v

    def _ids(self, market_id: str | None) -> list[str]:
        return [i for i, c in self._chunks.items() if market_id is None or c.market_id == market_id]

    def search(
        self, vector: np.ndarray, k: int = 8, market_id: str | None = None
    ) -> list[ScoredChunk]:
        ids = self._ids(market_id)
        if not ids:
            return []
        mat = np.stack([self._vecs[i] for i in ids])
        q = vector / (np.linalg.norm(vector) or 1.0)
        scores = mat @ q
        order = np.argsort(-scores)[:k]
        return [ScoredChunk(chunk=self._chunks[ids[j]], score=float(scores[j])) for j in order]

    def all_chunks(self, market_id: str | None = None) -> list[Chunk]:
        return [self._chunks[i] for i in self._ids(market_id)]

    def count(self, market_id: str | None = None) -> int:
        return len(self._ids(market_id))


class QdrantVectorStore:
    """Qdrant-backed store. Payload carries the chunk; ``market_id`` is indexed for filtering."""

    def __init__(
        self,
        url: str = ":memory:",
        collection: str = "pm_evidence",
        dim: int = 512,
        api_key: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif url.startswith("http"):
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            if url.startswith("http"):
                self.client.create_payload_index(
                    collection, "market_id", field_schema=models.PayloadSchemaType.KEYWORD
                )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _filter(self, market_id: str | None) -> Any:
        if market_id is None:
            return None
        m = self._models
        return m.Filter(
            must=[m.FieldCondition(key="market_id", match=m.MatchValue(value=market_id))]
        )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        points = [
            self._models.PointStruct(
                id=self._point_id(c.id), vector=v.tolist(), payload=c.model_dump(mode="json")
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def search(
        self, vector: np.ndarray, k: int = 8, market_id: str | None = None
    ) -> list[ScoredChunk]:
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            query_filter=self._filter(market_id),
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(p.payload), score=float(p.score))
            for p in res.points
        ]

    def all_chunks(self, market_id: str | None = None) -> list[Chunk]:
        out: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=self._filter(market_id),
                limit=256,
                offset=offset,
                with_payload=True,
            )
            out.extend(Chunk.model_validate(p.payload) for p in points)
            if offset is None:
                return out

    def count(self, market_id: str | None = None) -> int:
        return int(
            self.client.count(
                self.collection, count_filter=self._filter(market_id), exact=True
            ).count
        )


def build_store(settings: Settings, dim: int) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    return QdrantVectorStore(
        settings.qdrant_url, settings.collection, dim=dim, api_key=settings.qdrant_api_key
    )
