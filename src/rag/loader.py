"""PDF loading: extract text page by page so every chunk knows its page number."""
import re
from pathlib import Path

from pypdf import PdfReader


def _clean(text: str) -> str:
    """Normalise whitespace and re-join words hyphenated across line breaks."""
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[dict]:
    """Return one record per non-empty page: {doc, page (1-based), text}."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append({"doc": path.name, "page": i, "text": text})
    return pages


def load_folder(folder: Path) -> list[dict]:
    """Load every PDF in a folder (non-recursive), sorted for determinism."""
    pages = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        pages.extend(load_pdf(pdf))
    return pages
