"""Sentence-aware sliding-window chunking."""

from __future__ import annotations

import re

from .models import Chunk, Document

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Pack sentences into chunks of <= ``chunk_size`` chars with ~``overlap`` chars carried over.

    Sentences longer than ``chunk_size`` are hard-wrapped.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = " ".join(text.split())
    if not text:
        return []
    sentences: list[str] = []
    for s in _SENT.split(text):
        while len(s) > chunk_size:
            sentences.append(s[:chunk_size])
            s = s[chunk_size - overlap :]
        if s:
            sentences.append(s)

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for s in sentences:
        if current and length + len(s) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # carry trailing sentences for overlap
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            current, length = carry, carry_len
        current.append(s)
        length += len(s) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    body = doc.text if doc.text.startswith(doc.title) else f"{doc.title}. {doc.text}"
    return [
        Chunk(
            id=f"{doc.id}:{i}",
            doc_id=doc.id,
            text=piece,
            title=doc.title,
            url=doc.url,
            source=doc.source,
            published_at=doc.published_at,
            market_id=doc.market_id,
        )
        for i, piece in enumerate(split_text(body, chunk_size, overlap))
    ]
