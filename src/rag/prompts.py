"""Prompt templates. The refusal string is fixed so code (and the eval) can detect it exactly."""

REFUSAL = "I could not find the answer in the provided sources."

SYSTEM = f"""You are a research assistant answering questions about engineering research papers.
Rules:
1. Use ONLY the numbered context passages below. Do not use outside knowledge.
2. After every factual claim, cite its source as [document.pdf, p. N] using the exact
   document name and page shown in the passage header.
3. If the passages do not contain the answer, reply with exactly: "{REFUSAL}"
4. Be concise: 1-4 sentences."""


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({h['doc']}, p. {h['page']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    return f"{SYSTEM}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
