"""Streamlit chat UI.  Run: streamlit run app.py   (add `-- --mock` for no API calls)"""
import sys

import streamlit as st

from src.rag.config import settings
from src.rag.llm import LLM
from src.rag.pipeline import RAG, ingest
from src.rag.store import VectorStore

MOCK = "--mock" in sys.argv

st.set_page_config(page_title="Research Paper RAG", page_icon="📄")


@st.cache_resource
def get_rag() -> RAG:
    store = VectorStore()
    ingest(settings.papers_dir, store)  # index sample papers on first launch
    return RAG(store, LLM("mock" if MOCK else None))


rag = get_rag()

# ---- Sidebar: upload + index status ----
with st.sidebar:
    st.header("Documents")
    st.caption(f"LLM: {rag.llm.provider} / {rag.llm.model}")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index uploads"):
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            (settings.uploads_dir / f.name).write_bytes(f.getbuffer())
        with st.spinner("Embedding..."):
            added = ingest(settings.uploads_dir, rag.store)
        st.success(f"Indexed {len(added)} new file(s)" if added else "Already indexed")
    st.subheader("Indexed")
    for d in sorted(rag.store.indexed_docs()):
        st.write(f"• {d}")

# ---- Chat ----
st.title("📄 Research Paper Assistant")
st.caption("Answers come only from your PDFs, with [document, page] citations.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if q := st.chat_input("Ask about the papers..."):
    st.chat_message("user").markdown(q)
    with st.spinner("Searching..."):
        res = rag.answer(q)
    body = res["answer"]
    with st.chat_message("assistant"):
        st.markdown(body)
        if res["hits"] and not res["refused"]:
            with st.expander("Sources"):
                for h in res["hits"]:
                    st.markdown(f"**{h['doc']} p.{h['page']}** (sim {h['score']:.2f})\n\n> {h['text'][:400]}...")
    st.session_state.history += [("user", q), ("assistant", body)]
