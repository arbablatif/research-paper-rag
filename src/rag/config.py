"""Central configuration. Everything tunable lives here and can be overridden via .env."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    # Paths
    papers_dir: Path = ROOT / "data" / "papers"
    uploads_dir: Path = ROOT / "data" / "uploads"
    chroma_dir: Path = ROOT / "chroma_db"
    cache_dir: Path = ROOT / ".cache" / "llm"

    # Chunking (characters). Small chunks = precise page citations.
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    # Cosine similarity below which we refuse without calling the LLM.
    min_similarity: float = 0.25

    # LLM provider: gemini | anthropic | openai | mock
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5-mini"))


settings = Settings()
