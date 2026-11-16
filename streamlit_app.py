"""
Streamlit CHATBOT for the Indian Civil Law RAG.

Features: chat with memory (follow-up questions), streaming answers, sources for every answer,
example questions, upload PDFs into the FAISS vector DB, retrieval settings,
download chat, and thumbs up/down feedback sent to LangSmith.

Run:
    streamlit run streamlit_app.py
"""
import os
import uuid
from datetime import datetime

import streamlit as st

from config import (BM25_WEIGHT, CHAT_MODEL, EMBED_MODEL, INDEX_DIR, TOP_K,
                    get_embeddings, index_problem, langsmith_enabled, langsmith_status)
import ingest
from app import CivilLawRAG, source_name

st.set_page_config(page_title="Indian Civil Law Chatbot", page_icon="⚖️", layout="centered")

AVATARS = {"user": "👤", "assistant": "⚖️"}
EXAMPLES = [
    "My neighbour built a wall 2 feet inside my plot. What can I do?",
    "I paid advance for a house but the seller refuses to register the sale deed.",
    "My friend took a hand loan and the cheque he gave bounced.",
    "My brothers are not giving my sister her share in ancestral property.",
    "What is the BNS section for IPC 420 and IPC 406?",
    "My tenant is not vacating after the lease ended.",
]


# ---------------- Loaded once and reused (cached) ----------------
@st.cache_resource(show_spinner="Loading Hugging Face embedding model...")
def load_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Loading FAISS vector DB...")
def load_rag():
    return CivilLawRAG(embeddings=load_embeddings())


# ---------------- Helpers ----------------
def to_sources(docs):
    """Turn retrieved Documents into small dicts we can keep in the chat history."""
    sources = []
    for d in docs:
        text = d.page_content.strip()
        sources.append({
            "file": source_name(d),
            "title": text.splitlines()[0].lstrip("# ").strip()[:90],
            "text": text[:500] + ("..." if len(text) > 500 else ""),
        })
    return sources


def send_feedback(run_id, key):
    """Called when the user clicks 👍 / 👎 -> stored on the LangSmith trace of that answer."""
    score = st.session_state.get(key)   # 1 = thumbs up, 0 = thumbs down
    if score is None:
        return
    try:
        from langsmith import Client
        Client().create_feedback(run_id=run_id, key="user-thumbs", score=score)
        st.toast("Thanks! Feedback saved in LangSmith.")
    except Exception as e:
        st.toast(f"Could not send feedback: {e}")


def show_extras(msg):
    """Sources expander + feedback buttons under an assistant answer."""
    if msg.get("sources"):
        with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
            for i, s in enumerate(msg["sources"], 1):
                st.markdown(f"**{i}. {s['file']}** — {s['title']}")
                st.caption(s["text"])
    if msg.get("run_id") and langsmith_enabled():
        key = f"fb_{msg['run_id']}"
        st.feedback("thumbs", key=key, on_change=send_feedback, args=(msg["run_id"], key))


def save_uploads(files):
    """Save uploaded files into data/uploads/. Skip names that are already there."""
    ingest.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for f in files:
        path = ingest.UPLOAD_DIR / os.path.basename(f.name)
        if path.exists():
            skipped.append(f.name)
            continue
        path.write_bytes(f.getbuffer())
        saved.append(path)
    return saved, skipped


def chat_as_markdown(messages):
    lines = [f"# Indian Civil Law Chat - {datetime.now():%d %b %Y, %H:%M}\n"]
    for m in messages:
        lines.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}:**\n\n{m['content']}\n")
    return "\n".join(lines)


if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚖️ Civil Law Chatbot")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.download_button("⬇️ Download chat", chat_as_markdown(st.session_state.messages),
                       file_name="civil_law_chat.md", mime="text/markdown",
                       disabled=not st.session_state.messages, use_container_width=True)

    st.subheader("🔎 Retrieval settings")
    top_k = st.slider("Chunks per retriever", 2, 10, TOP_K,
                      help="How many chunks BM25 and FAISS each return (merged afterwards).")
    bm25_weight = st.slider("BM25 keyword weight", 0.0, 1.0, BM25_WEIGHT, 0.1,
                            help="0 = only FAISS (meaning), 1 = only BM25 (exact words like 'Section 420').")

    st.subheader("📄 Add documents")
    files = st.file_uploader("Bare Acts, judgments, notes", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    if st.button("Add to vector DB", disabled=not files, use_container_width=True):
        saved, skipped = save_uploads(files)
        if saved:
            with st.spinner(f"Embedding {len(saved)} file(s) with {EMBED_MODEL}..."):
                try:
                    n = ingest.add_files(saved, embeddings=load_embeddings())
                except Exception as e:
                    n = None
                    st.error(str(e))
            if n:
                load_rag.clear()   # reload so BM25 + FAISS include the new chunks
                st.success(f"Added {n} chunks from {len(saved)} file(s).")
            else:
                for p in saved:
                    p.unlink(missing_ok=True)
                if n == 0:
                    st.warning("No text found. Scanned PDF? OCR it first (e.g. ocrmypdf).")
        if skipped:
            st.info("Already added (skipped): " + ", ".join(skipped))

    with st.expander("Rebuild whole DB from data/ folder"):
        st.caption("Use after copying many files into data/ or changing EMBED_MODEL.")
        if st.button("Rebuild now", use_container_width=True):
            with st.spinner("Re-embedding all documents..."):
                n_docs, n_chunks = ingest.rebuild_all(load_embeddings())
            load_rag.clear()
            st.success(f"Rebuilt: {n_docs} documents/pages -> {n_chunks} chunks.")

    st.subheader("ℹ️ System")
    st.caption(f"**Vector DB:** `{INDEX_DIR}`  \n**Embeddings:** {EMBED_MODEL} (Hugging Face, local)  \n"
               f"**LLM:** {CHAT_MODEL}  \n{langsmith_status()}")

# ---------------- Main chat area ----------------
st.title("⚖️ Indian Civil Law Assistant")
st.caption("Describe your situation - property, money, contracts, tenancy, family, limitation - "
           "and get the civil remedies plus related IPC ↔ BNS sections. General information, not legal advice.")

problem = index_problem()
if problem:
    st.warning(problem)
    st.stop()
openai_key = os.getenv("OPENAI_API_KEY", "")
if not openai_key or openai_key.startswith("sk-your"):
    st.error("Add your OPENAI_API_KEY to the .env file and restart.")
    st.stop()

rag = load_rag()

# Show the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_extras(msg)

# Example questions for an empty chat
if not st.session_state.messages:
    st.markdown(f"**Try an example** · {rag.num_chunks} chunks in your vector DB")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

question = st.chat_input("Describe your situation or ask a legal question...")
question = question or st.session_state.pop("pending_question", None)

if question:
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    run_id = str(uuid.uuid4())   # same id as the LangSmith trace -> used for feedback
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        docs = []
        try:
            with st.status("Searching laws and judgments...") as status:
                stream = rag.chat(question, history, top_k=top_k, bm25_weight=bm25_weight,
                                  langsmith_extra={"run_id": run_id})
                docs = next(stream)          # 1st item = retrieved chunks
                status.update(label=f"Found {len(docs)} relevant chunks", state="complete")
            answer = st.write_stream(stream)  # then the answer streams in, token by token
        except Exception as e:
            answer = f"⚠️ Something went wrong: {e}"
            st.error(answer)
        msg = {"role": "assistant", "content": answer, "sources": to_sources(docs), "run_id": run_id}
        show_extras(msg)
    st.session_state.messages.append(msg)
