"""End-to-end RAG: ingest PDFs -> retrieve -> grounded answer with verified citations."""
import re
from pathlib import Path

from .chunker import chunk_pages
from .config import settings
from .llm import LLM
from .loader import load_pdf
from .prompts import REFUSAL, build_prompt
from .store import VectorStore

# Matches [doc.pdf, p. 3], [doc.pdf, page 3], [doc.pdf, pp. 2, 3] and [doc.pdf, pp. 2-3].
CITATION_RE = re.compile(r"\[([^\[\]]+?\.pdf),\s*p(?:age|p)?s?\.?\s*([\d,\s\-–]+)\]", re.I)


def ingest(folder: Path, store: VectorStore, force: bool = False) -> dict:
    """Index every PDF in `folder` that is not already in the store."""
    done = set() if force else store.indexed_docs()
    added = {}
    for pdf in sorted(Path(folder).glob("*.pdf")):
        if pdf.name in done:
            continue
        chunks = chunk_pages(load_pdf(pdf), settings.chunk_size, settings.chunk_overlap)
        store.add(chunks)
        added[pdf.name] = len(chunks)
    return added


def parse_citations(answer: str) -> list[dict]:
    """Extract [doc.pdf, p. N] citations from the model's answer."""
    out = []
    for doc, pages in CITATION_RE.findall(answer):
        nums = [int(n) for n in re.findall(r"\d+", pages)]
        if re.search(r"[\-–]", pages) and len(nums) == 2:  # a range like 2-4
            nums = list(range(nums[0], nums[1] + 1))
        out += [{"doc": doc.strip(), "page": n} for n in nums]
    return out


class RAG:
    def __init__(self, store: VectorStore | None = None, llm: LLM | None = None):
        self.store = store or VectorStore()
        self.llm = llm or LLM()

    def answer(self, question: str) -> dict:
        hits = self.store.query(question, settings.top_k)
        # Gate 1: retrieval. If nothing is relevant enough, refuse without spending an API call.
        if not hits or hits[0]["score"] < settings.min_similarity:
            return {"answer": REFUSAL, "refused": True, "citations": [], "hits": hits, "gated": True}

        text = self.llm.generate(build_prompt(question, hits))
        # Gate 2: the model itself was told to emit the exact refusal string.
        refused = REFUSAL.lower().rstrip(".") in text.lower()
        cites = parse_citations(text)
        # Verify each citation points at a passage we actually showed the model.
        shown = {(h["doc"], h["page"]) for h in hits}
        for c in cites:
            c["grounded"] = (c["doc"], c["page"]) in shown
        return {"answer": text, "refused": refused, "citations": cites, "hits": hits, "gated": False}
