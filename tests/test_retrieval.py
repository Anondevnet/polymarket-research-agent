import numpy as np
import pytest

from pm_research.chunking import chunk_document
from pm_research.embeddings import HashingEmbedder
from pm_research.models import Document
from pm_research.retrieval import BM25Index, Retriever, rrf
from pm_research.vectorstore import InMemoryVectorStore, QdrantVectorStore

DOCS = [
    Document(
        id="a",
        source="t",
        title="Fed cuts interest rates",
        text="The Fed lowered rates.",
        market_id="m1",
    ),
    Document(
        id="b", source="t", title="Bitcoin hits record", text="Crypto rallies.", market_id="m1"
    ),
    Document(
        id="c",
        source="t",
        title="Fed rate decision looms",
        text="Markets await the Fed.",
        market_id="m2",
    ),
]


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder(128)
    v1, v2 = e.embed(["hello world", "hello world"])
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0)


@pytest.mark.parametrize(
    "store_factory", [InMemoryVectorStore, lambda: QdrantVectorStore(":memory:", "t", dim=128)]
)
def test_vector_store_roundtrip_and_filter(store_factory) -> None:  # type: ignore[no-untyped-def]
    store = store_factory()
    e = HashingEmbedder(128)
    chunks = [c for d in DOCS for c in chunk_document(d)]
    store.upsert(chunks, e.embed([c.text for c in chunks]))
    assert store.count() == 3
    assert store.count("m1") == 2
    hits = store.search(e.embed(["fed interest rates"])[0], k=2, market_id="m1")
    assert hits[0].chunk.doc_id == "a"
    assert all(h.chunk.market_id == "m1" for h in hits)
    assert {c.doc_id for c in store.all_chunks("m2")} == {"c"}


def test_bm25_ranks_exact_terms() -> None:
    idx = BM25Index()
    idx.add([c for d in DOCS for c in chunk_document(d)])
    assert idx.search("bitcoin record")[0].chunk.doc_id == "b"
    assert idx.search("nonexistent") == []


def test_rrf_merges_lists() -> None:
    idx = BM25Index()
    chunks = [c for d in DOCS for c in chunk_document(d)]
    idx.add(chunks)
    a = idx.search("fed")
    fused = rrf([a, list(reversed(a))], k=2)
    assert len(fused) == 2 and fused[0].retriever == "hybrid"


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_retriever_modes(mode: str) -> None:
    r = Retriever(HashingEmbedder(128), InMemoryVectorStore())
    assert r.add_documents(DOCS) == 3
    hits = r.search("Fed interest rates", k=2, market_id="m1", mode=mode)
    assert hits and hits[0].chunk.doc_id == "a"


def test_retriever_rebuilds_bm25_from_persistent_store() -> None:
    store = InMemoryVectorStore()
    Retriever(HashingEmbedder(128), store).add_documents(DOCS)
    fresh = Retriever(HashingEmbedder(128), store)  # new process attaching to same collection
    assert fresh.search("bitcoin", k=1, mode="bm25")[0].chunk.doc_id == "b"
