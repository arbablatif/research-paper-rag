"""Local embeddings via sentence-transformers — no API calls, runs on CPU."""
from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer  # heavy import, load lazily
    return SentenceTransformer(settings.embed_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Return L2-normalised vectors so dot product == cosine similarity."""
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
