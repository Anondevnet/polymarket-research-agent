"""Retrieval eval: recall@k and MRR of dense / BM25 / hybrid on a small labelled set."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..embeddings import Embedder, HashingEmbedder
from ..models import Document
from ..retrieval import Retriever
from ..vectorstore import InMemoryVectorStore, VectorStore
from .metrics import mrr, recall_at_k


def _data_dir() -> Path:
    """$PMRA_EVAL_DATA, else ./data/eval, else the repo checkout (editable install)."""
    if env := os.getenv("PMRA_EVAL_DATA"):
        return Path(env)
    cwd = Path.cwd() / "data" / "eval"
    return cwd if cwd.exists() else Path(__file__).resolve().parents[3] / "data" / "eval"


DATA = _data_dir()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_retrieval_eval(
    corpus_path: Path = DATA / "retrieval_corpus.jsonl",
    queries_path: Path = DATA / "retrieval_queries.jsonl",
    ks: tuple[int, ...] = (1, 3, 5),
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> dict[str, Any]:
    embedder = embedder or HashingEmbedder(512)
    retriever = Retriever(embedder, store or InMemoryVectorStore(), hybrid=True)
    docs = [Document.model_validate(d) for d in load_jsonl(corpus_path)]
    retriever.add_documents(docs)
    queries = load_jsonl(queries_path)
    results: dict[str, Any] = {
        "embedder": embedder.name,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "modes": {},
    }
    for mode in ("bm25", "dense", "hybrid"):
        agg = {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0}
        for q in queries:
            hits = retriever.search(q["query"], k=max(ks), mode=mode)
            doc_ids = list(dict.fromkeys(h.chunk.doc_id for h in hits))
            rel = set(q["relevant"])
            for k in ks:
                agg[f"recall@{k}"] += recall_at_k(doc_ids, rel, k)
            agg["mrr"] += mrr(doc_ids, rel)
        results["modes"][mode] = {m: round(v / len(queries), 4) for m, v in agg.items()}
    return results
