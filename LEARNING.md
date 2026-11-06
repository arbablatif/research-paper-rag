# LEARNING.md: How this works, in plain language

## 1. What is RAG?

**Retrieval-Augmented Generation** means: *look it up first, then answer.*
A language model on its own answers from what it memorized during training. That memory can be outdated or wrong, and the model can't say where a fact came from. In RAG we:
1. **Retrieve** the few passages from *your* documents that are most relevant to the question.
2. **Generate** an answer while instructing the model to use *only* those passages.

> Analogy: an open-book exam. The model is the student, and the retriever hands it the right pages of the book.

## 2. Why chunk, and why by page?

A 15-page paper is too long to send whole for every question, and it would dilute relevance. So we cut each paper into ~900-character **chunks**, about a paragraph or two each.
- **Overlap (150 characters):** a sentence cut at a boundary still appears whole in the next chunk.
- **Never cross a page:** each chunk belongs to exactly one page, so the citation `[paper.pdf, p. 7]` is always exact. This is the simplest way to get trustworthy page citations.

Trade-off: smaller chunks give more precise citations but less context per chunk. 900 characters with top-5 retrieval is a common middle ground.

## 3. Embeddings and vector search

An **embedding** turns text into a list of numbers (384 here). Texts with similar meanings land close together in that space, even when they use different words ("learning rate warm-up" ≈ "warmup_steps").
- We use **sentence-transformers (MiniLM)** running **locally**: free, private, and no API calls.
- **Chroma** is the vector database. It stores every chunk's vector along with its document and page, and it quickly finds the chunks closest to the question's vector.
- **Cosine similarity** (from 0 to 1 here) measures closeness. Higher means more relevant.

## 4. How refusal works (the anti-hallucination design)

There are two layers:
1. **Retrieval gate:** if even the best chunk scores below 0.25, the question clearly isn't about these papers. We refuse immediately, which **costs nothing** because the LLM is never called.
2. **Prompt rule:** the model is told that if the passages don't contain the answer, it must reply with the exact sentence *"I could not find the answer in the provided sources."* Because the sentence is fixed, code can detect a refusal reliably.

Why both? The gate catches obvious off-topic questions cheaply. The prompt rule catches subtle cases, such as a question about a *related* paper that isn't in the folder (like the ViT and GPT-3 questions in the eval).

## 5. How citations are checked

The model writes citations like `[resnet.pdf, p. 2]`. We **parse** them with a regular expression and **verify** that each one points to a passage we actually showed the model. A citation that fails this check is a red flag that the model made it up. This is the "citation grounding" metric.

## 6. Caching and mock mode (keeping costs near zero)

- **Disk cache:** every LLM response is saved under `.cache/llm/`, keyed by a hash of (provider + model + prompt). Asking the same thing twice costs nothing, and rerunning the whole evaluation is free.
- **`--mock`:** a fake LLM that simply quotes the top passage. The entire system (UI, evaluation, tests) runs with **no API key and no cost**. That's useful for demos, CI, and development.

## 7. Swappable LLM providers

`llm.py` hides the provider behind one `generate(prompt)` function. Switching between Gemini (the default, and cheapest), Anthropic, and OpenAI takes one line in `.env`, with no code changes. Each SDK is imported only when it's used.

## 8. How the evaluation works

- **15 questions:** 12 that the papers answer, and 3 that they don't.
- **Answer accuracy:** the answer must contain the key facts (e.g. "4000" for warm-up steps). This is simple string matching, which is cheap and repeatable. We deliberately avoided an "LLM-as-judge" because it would double the API calls and add its own errors.
- **Citation correctness:** the answer must cite the right document and a page within ±1 of where the fact actually appears.
- **Refusal accuracy:** for the 3 unanswerable questions, did it refuse?
- **Mock baseline:** shows how much comes from retrieval alone versus the LLM.

**Result:** about 83% answer and citation accuracy and 100% correct refusals. Both misses were *retrieval* problems, not the model inventing things. In one case the model correctly refused when it didn't get the right page. In a client pitch, that's the key point: **when this system fails, it fails safely.**

## 9. What I'd improve next (good talking points)

- **Hybrid search (BM25 + vectors):** exact keyword matches would help with formula-heavy pages like the Adam algorithm box.
- **Reranker:** a cross-encoder re-scores the top 20 candidates for better precision.
- **Better PDF parsing:** layout-aware tools would preserve tables and equations.
- **Bigger eval set:** with an LLM judge for paraphrased answers.
