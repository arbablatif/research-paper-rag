"""Unit tests for page-aware chunking and citation parsing (no models, no API)."""
from src.rag.chunker import chunk_pages, chunk_text
from src.rag.pipeline import parse_citations


def test_chunks_respect_size_and_cover_text():
    text = "Sentence number one. " * 200
    chunks = chunk_text(text, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("Sentence") and chunks[-1].endswith("one.")


def test_chunks_never_cross_pages():
    pages = [{"doc": "a.pdf", "page": 1, "text": "alpha " * 100},
             {"doc": "a.pdf", "page": 2, "text": "beta " * 100}]
    chunks = chunk_pages(pages, size=200, overlap=20)
    assert all(("alpha" in c["text"]) != ("beta" in c["text"]) for c in chunks)
    assert {c["page"] for c in chunks} == {1, 2}
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_parse_citations_variants():
    ans = "X [resnet.pdf, p. 3] and Y [adam_optimizer.pdf, p.2] and [wavenet.pdf, page 5]."
    assert parse_citations(ans) == [
        {"doc": "resnet.pdf", "page": 3},
        {"doc": "adam_optimizer.pdf", "page": 2},
        {"doc": "wavenet.pdf", "page": 5},
    ]


def test_parse_multi_page_citations():
    assert parse_citations("[resnet.pdf, p. 2, 3]") == [
        {"doc": "resnet.pdf", "page": 2}, {"doc": "resnet.pdf", "page": 3}]
    assert [c["page"] for c in parse_citations("[wavenet.pdf, pp. 3-5]")] == [3, 4, 5]
