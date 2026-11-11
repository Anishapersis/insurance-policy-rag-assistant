"""Streamlit chat UI for the policy assistant.  Run: streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag import HybridRetriever, answer, load_chunks  # noqa: E402

st.set_page_config(page_title="Policy Q&A Assistant", page_icon="📄")


@st.cache_resource
def get_retriever():
    return HybridRetriever(load_chunks())


retriever = get_retriever()
st.title("Auto policy Q&A assistant")
st.caption("Retrieval-augmented answers with citations. Hybrid BM25 + dense retrieval, fused with RRF. "
           "Documents are a fictional sample policy.")

with st.sidebar:
    k = st.slider("Passages to retrieve", 1, 8, 3)
    st.markdown("**Try:**\n- How much do I pay if a deer hits my car?\n- Am I covered driving for Uber?\n"
                "- What is the deadline for a Proof of Loss?\n- Will my rates go up if someone rear-ends me?")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

if q := st.chat_input("Ask about the policy..."):
    st.chat_message("user").markdown(q)
    hits = retriever.search(q, k=k)
    out = answer(q, hits)
    body = out["answer"]
    if out["sources"]:
        body += "\n\n**Sources:** " + "; ".join(dict.fromkeys(out["sources"]))
    body += f"\n\n<sub>mode: {out['mode']}</sub>"
    st.chat_message("assistant").markdown(body, unsafe_allow_html=True)
    with st.expander("Retrieved passages"):
        for i, (c, s) in enumerate(hits, 1):
            st.markdown(f"**[{i}] {c.citation}** (score {s:.3f})\n\n{c.text}")
    st.session_state.history += [("user", q), ("assistant", body)]
