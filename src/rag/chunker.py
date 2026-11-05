"""Page-aware chunking. A chunk never crosses a page boundary, so its citation is exact."""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~size-char windows with overlap, preferring sentence/space breaks."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Back off to the last sentence end (or space) in the second half of the window.
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut < size // 2:
                cut = window.rfind(" ")
            if cut > size // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """Turn page records into chunk records that carry doc + page + a stable id."""
    out = []
    for p in pages:
        for j, piece in enumerate(chunk_text(p["text"], size, overlap)):
            out.append({
                "id": f"{p['doc']}::p{p['page']}::c{j}",
                "doc": p["doc"],
                "page": p["page"],
                "text": piece,
            })
    return out
