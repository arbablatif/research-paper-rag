"""End-to-end smoke test in mock mode: real PDFs + local embeddings, zero API calls."""
from src.rag.config import settings
from src.rag.llm import LLM
from src.rag.pipeline import RAG, ingest
from src.rag.prompts import REFUSAL
from src.rag.store import VectorStore


def test_mock_pipeline(tmp_path):
    store = VectorStore(path=tmp_path / "db")
    added = ingest(settings.papers_dir, store)
    assert len(added) >= 4
    rag = RAG(store, LLM("mock"))

    res = rag.answer("What is scaled dot-product attention?")
    assert not res["refused"]
    assert res["citations"] and all(c["grounded"] for c in res["citations"])

    off = rag.answer("What is the best recipe for banana bread?")
    assert off["refused"] and off["answer"] == REFUSAL
