"""CLI: index the PDFs in data/papers into Chroma.  Usage: python ingest.py [--reset]"""
import argparse

from src.rag.config import settings
from src.rag.pipeline import ingest
from src.rag.store import VectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop the index and rebuild")
    args = ap.parse_args()
    store = VectorStore()
    if args.reset:
        store.reset()
    added = ingest(settings.papers_dir, store)
    for doc, n in added.items():
        print(f"indexed {doc}: {n} chunks")
    print(f"total chunks in store: {store.col.count()}")


if __name__ == "__main__":
    main()
