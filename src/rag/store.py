"""Chroma vector store wrapper: add chunks, query by similarity, track indexed docs."""
import chromadb

from .config import settings
from .embedder import embed

COLLECTION = "papers"


class VectorStore:
    def __init__(self, path=None):
        self.client = chromadb.PersistentClient(path=str(path or settings.chroma_dir))
        # Cosine space so distance = 1 - cosine similarity.
        self.col = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def indexed_docs(self) -> set[str]:
        metas = self.col.get(include=["metadatas"])["metadatas"]
        return {m["doc"] for m in metas}

    def add(self, chunks: list[dict], batch: int = 128) -> None:
        for i in range(0, len(chunks), batch):
            b = chunks[i:i + batch]
            self.col.upsert(
                ids=[c["id"] for c in b],
                documents=[c["text"] for c in b],
                embeddings=embed([c["text"] for c in b]),
                metadatas=[{"doc": c["doc"], "page": c["page"]} for c in b],
            )

    def query(self, question: str, k: int) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(query_embeddings=embed([question]), n_results=min(k, self.col.count()))
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"text": text, "doc": meta["doc"], "page": meta["page"], "score": 1 - dist})
        return hits

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION)
        self.col = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
