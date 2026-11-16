"""
RAG core + terminal chat for the Indian Civil Law assistant.

Retrieval = Hybrid search
    BM25  (keyword search - great for "Section 420", "Order 39", "Article 65")
  + FAISS (semantic search with Hugging Face embeddings - understands your situation)
Both results are merged by LangChain's EnsembleRetriever, then OpenAI writes the answer.
Follow-up questions ("what is the time limit for that?") are rewritten using chat history.
Every question is traced in LangSmith (rewrite, retrieved chunks, prompt, answer, tokens, latency).

Run the terminal chat:   python app.py
Run the web chatbot:     streamlit run streamlit_app.py
"""
import re

from config import (BM25_WEIGHT, CHAT_MODEL, EMBED_MODEL, INDEX_DIR, TOP_K,
                    get_embeddings, index_problem, langsmith_status)  # loads .env first

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langsmith import traceable

HISTORY_TURNS = 4        # last N chat messages sent to the LLM as memory
HISTORY_CHARS = 1500     # each old message is cut to this length to save tokens

SYSTEM_PROMPT = """You are an Indian civil-law research assistant (chatbot).
The user describes a real-life situation (use case) or asks a legal question.
Answer ONLY from the CONTEXT. If the context does not cover something, say
"Not found in my documents" for that part - never invent sections or case names.

For a NEW situation or legal question, use this format (skip a heading only if it does not apply):

1. **Issue in brief** - what the legal problem is, in 1-2 lines.
2. **Applicable civil law** - Acts and sections (CPC, Specific Relief Act, Limitation Act,
   Transfer of Property Act, Contract Act, Hindu Succession Act, etc.).
3. **Related criminal provisions (IPC -> BNS)** - give BOTH the old IPC section and the new BNS
   section. Remind that BNS applies to acts on/after 1 July 2024, IPC to earlier acts.
   Say clearly if the matter is purely civil.
4. **Remedies & practical steps** - what to file, in which court/forum, interim relief.
5. **Limitation period** - time limit and the Limitation Act article/section.
6. **Relevant case law** - judgments from the context with citation.
7. **Sources** - file names from the context you used.

For a short FOLLOW-UP question, answer directly and briefly (no full format), still citing sections.
If the user only greets you, reply in one line and ask them to describe their legal situation.

End every legal answer with: "_This is general legal information, not legal advice. Please consult an advocate._"
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "CONTEXT:\n{context}\n\nUSER QUESTION / USE CASE:\n{question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest message as a standalone search query about Indian law. "
               "Use the chat history to replace words like 'it', 'that', 'this case' with what they "
               "refer to. Return ONLY the rewritten query. If it is already standalone, return it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def bm25_tokenizer(text):
    """Lower-case word tokens so 'Section 420,' matches 'section 420'."""
    return re.findall(r"\w+", text.lower())


def source_name(doc):
    """'cpc_1908.pdf (p. 45)' for PDFs, file name for text files."""
    page = doc.metadata.get("page_label") or (doc.metadata["page"] + 1 if "page" in doc.metadata else None)
    return f"{doc.metadata.get('file', 'unknown')}" + (f" (p. {page})" if page else "")


def format_docs(docs):
    return "\n\n---\n\n".join(f"[Source: {source_name(d)}]\n{d.page_content}" for d in docs)


def trim_history(history):
    """history = list of (role, text) with role 'user'/'assistant'. Keep it short."""
    return [(role, text[:HISTORY_CHARS]) for role, text in (history or [])[-HISTORY_TURNS:]]


def _trace_output(outputs):
    """How the streamed output is shown in LangSmith: sources + full answer text."""
    docs = outputs[0] if outputs and isinstance(outputs[0], list) else []
    return {"answer": "".join(o for o in outputs if isinstance(o, str)),
            "sources": [d.metadata.get("file") for d in docs]}


class CivilLawRAG:
    def __init__(self, embeddings=None, llm=None):
        problem = index_problem()
        if problem:
            raise RuntimeError(problem)

        embeddings = embeddings or get_embeddings()
        # allow_dangerous_deserialization=True is safe: you created this index yourself with ingest.py
        vectordb = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        all_chunks = list(vectordb.docstore._dict.values())
        self.num_chunks = len(all_chunks)

        # Hybrid retriever: BM25 is built from the same chunks stored in the FAISS docstore
        self.faiss = vectordb.as_retriever(search_kwargs={"k": TOP_K})
        self.bm25 = BM25Retriever.from_documents(all_chunks, preprocess_func=bm25_tokenizer, k=TOP_K)
        self.retriever = EnsembleRetriever(retrievers=[self.bm25, self.faiss],
                                           weights=[BM25_WEIGHT, 1 - BM25_WEIGHT])

        llm = llm or ChatOpenAI(model=CHAT_MODEL)
        self.answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        self.rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    @traceable(name="Indian Civil Law RAG", run_type="chain", reduce_fn=_trace_output,
               process_inputs=lambda inputs: {k: v for k, v in inputs.items() if k != "self"},
               metadata={"embed_model": EMBED_MODEL, "chat_model": CHAT_MODEL})
    def chat(self, question, history=None, top_k=TOP_K, bm25_weight=BM25_WEIGHT):
        """Generator: FIRST yields the list of retrieved docs, THEN the answer token by token."""
        history = trim_history(history)

        # 1) Follow-up question? Rewrite it into a standalone search query using the history
        query = question
        if history:
            query = self.rewrite_chain.invoke({"history": history, "question": question},
                                              config={"run_name": "Rewrite follow-up question"})

        # 2) Hybrid retrieval (BM25 + FAISS)
        self.bm25.k = top_k
        self.faiss.search_kwargs["k"] = top_k
        self.retriever.weights = [bm25_weight, 1 - bm25_weight]
        docs = self.retriever.invoke(query, config={"run_name": "Hybrid retrieval (BM25 + FAISS)"})
        yield docs

        # 3) Stream the answer
        yield from self.answer_chain.stream(
            {"history": history, "context": format_docs(docs), "question": question},
            config={"run_name": "Answer generation"},
        )

    def ask(self, question, history=None):
        """Non-streaming helper -> (answer, docs)."""
        stream = self.chat(question, history)
        docs = next(stream)
        return "".join(stream), docs


def main():
    try:
        rag = CivilLawRAG()
    except RuntimeError as e:
        raise SystemExit(str(e))

    print("Indian Civil Law RAG  (BM25 + FAISS)  - type 'exit' to quit, 'new' for a new chat")
    print(f"Vector DB : {INDEX_DIR}  ({rag.num_chunks} chunks)")
    print(f"Embeddings: {EMBED_MODEL} (Hugging Face, local)   LLM: {CHAT_MODEL}")
    print(langsmith_status())
    print("\nExample: My neighbour built a wall 2 feet inside my plot. What can I do?")

    history = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question.lower() == "new":
            history = []
            print("(new chat started)")
            continue
        if not question:
            continue

        stream = rag.chat(question, history)
        docs = next(stream)
        print("\nAssistant: ", end="")
        answer = ""
        for token in stream:
            print(token, end="", flush=True)
            answer += token
        print("\n\nRetrieved from:", ", ".join(sorted({d.metadata.get("file", "?") for d in docs})))
        history += [("user", question), ("assistant", answer)]


if __name__ == "__main__":
    main()
