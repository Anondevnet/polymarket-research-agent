"""Dense, sparse (BM25), and hybrid (reciprocal-rank fusion) retrieval."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .chunking import chunk_document
from .embeddings import Embedder, tokenize
from .models import Chunk, Document, ScoredChunk
from .tracing import span
from .vectorstore import VectorStore


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs: dict[str, tuple[Chunk, Counter[str], int]] = {}
        self._df: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.id in self._docs:
                continue
            toks = tokenize(c.text)
            tf = Counter(toks)
            self._docs[c.id] = (c, tf, len(toks))
            self._df.update(tf.keys())

    def search(self, query: str, k: int = 8, market_id: str | None = None) -> list[ScoredChunk]:
        if not self._docs:
            return []
        n = len(self._docs)
        avgdl = sum(d[2] for d in self._docs.values()) / n
        q = tokenize(query)
        scores: list[tuple[float, Chunk]] = []
        for chunk, tf, dl in self._docs.values():
            if market_id is not None and chunk.market_id != market_id:
                continue
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                f = tf[t]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scores.append((s, chunk))
        scores.sort(key=lambda x: -x[0])
        return [ScoredChunk(chunk=c, score=s, retriever="bm25") for s, c in scores[:k]]


def rrf(results: list[list[ScoredChunk]], k: int = 8, c: int = 60) -> list[ScoredChunk]:
    """Reciprocal-rank fusion of several ranked lists."""
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, Chunk] = {}
    for ranked in results:
        for rank, sc in enumerate(ranked):
            fused[sc.chunk.id] += 1.0 / (c + rank + 1)
            by_id[sc.chunk.id] = sc.chunk
    order = sorted(fused, key=lambda i: -fused[i])[:k]
    return [ScoredChunk(chunk=by_id[i], score=fused[i], retriever="hybrid") for i in order]


class Retriever:
    """Owns chunking + embedding on write and dense/BM25/hybrid search on read."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        hybrid: bool = True,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.hybrid = hybrid
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25 = BM25Index()
        self._bm25_synced = False

    def add_documents(self, docs: list[Document]) -> int:
        chunks = [c for d in docs for c in chunk_document(d, self.chunk_size, self.chunk_overlap)]
        if not chunks:
            return 0
        with span(
            "embed_and_upsert", as_type="embedding", n_chunks=len(chunks), model=self.embedder.name
        ):
            vecs = self.embedder.embed([c.text for c in chunks])
            self.store.upsert(chunks, vecs)
            self.bm25.add(chunks)
        return len(chunks)

    def _sync_bm25(self) -> None:
        # Rebuild the sparse index from the store when attaching to a persistent collection.
        if not self._bm25_synced:
            self.bm25.add(self.store.all_chunks())
            self._bm25_synced = True

    def search(
        self, query: str, k: int = 8, market_id: str | None = None, mode: str | None = None
    ) -> list[ScoredChunk]:
        mode = mode or ("hybrid" if self.hybrid else "dense")
        with span(
            "retrieve", as_type="retriever", input=query, k=k, mode=mode, market_id=market_id or ""
        ) as s:
            dense: list[ScoredChunk] = []
            if mode in {"dense", "hybrid"}:
                qv = self.embedder.embed([query])[0]
                dense = self.store.search(
                    qv, k=k * 2 if mode == "hybrid" else k, market_id=market_id
                )
            if mode == "dense":
                out = dense[:k]
            else:
                self._sync_bm25()
                sparse = self.bm25.search(query, k=k * 2, market_id=market_id)
                out = sparse[:k] if mode == "bm25" else rrf([dense, sparse], k=k)
            s.output([{"id": r.chunk.id, "score": round(r.score, 4)} for r in out])
            return out
