import pytest

from pm_research.chunking import chunk_document, split_text
from pm_research.models import Document


def test_short_text_single_chunk() -> None:
    assert split_text("One sentence. Two sentences.", 100, 10) == ["One sentence. Two sentences."]


def test_chunks_respect_size_and_overlap() -> None:
    text = " ".join(f"Sentence number {i} has some words." for i in range(60))
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share at least one sentence (overlap)
    assert any(s in chunks[1] for s in chunks[0].split(". ")[-2:])


def test_hard_wrap_long_sentence() -> None:
    chunks = split_text("x" * 1000, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)


def test_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_chunk_document_carries_metadata() -> None:
    doc = Document(id="d", source="s", title="Title", text="Body text.", url="u", market_id="m")
    (chunk,) = chunk_document(doc)
    assert chunk.id == "d:0" and chunk.market_id == "m" and chunk.text.startswith("Title.")
