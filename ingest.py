"""
STEP 1 - Build the local FAISS vector database.

Reads every .pdf / .txt / .md file inside ./data (any sub-folder),
splits it into chunks, creates embeddings with a Hugging Face model
(runs on your computer, no API cost) and saves the FAISS index to
./faiss_index inside the project folder.

Run again whenever you add new files to ./data or change EMBED_MODEL:
    python ingest.py
(Files uploaded from the Streamlit chatbot are added automatically - no rebuild needed.)
"""
import os
from pathlib import Path

from config import (DATA_DIR, EMBED_MODEL, INDEX_DIR, MODEL_FILE,
                    get_embeddings, index_problem)  # loads .env first

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_DIR = DATA_DIR / "uploads"   # files uploaded from the chatbot are stored here


def add_metadata(docs):
    """file name + category (= first sub-folder of data/, e.g. bare_acts / judgments / uploads)."""
    for d in docs:
        rel_path = os.path.relpath(d.metadata.get("source", ""), DATA_DIR)
        parts = rel_path.split(os.sep)
        d.metadata["file"] = parts[-1]
        d.metadata["category"] = parts[0] if len(parts) > 1 else "general"
    return docs


def load_documents():
    """Load all PDFs, text and markdown files from DATA_DIR."""
    loaders = [
        ("**/*.pdf", PyPDFLoader, {}),
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
    ]
    docs = []
    for pattern, loader_cls, kwargs in loaders:
        loader = DirectoryLoader(
            str(DATA_DIR), glob=pattern, loader_cls=loader_cls,
            loader_kwargs=kwargs, silent_errors=True,
        )
        docs.extend(loader.load())
    return add_metadata(docs)


def split_documents(docs):
    """Split into overlapping chunks. '### ' headings keep one section per chunk where possible."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if c.page_content.strip()]  # drop empty (e.g. scanned PDF pages)


def build_index(chunks, embeddings=None):
    """Embed chunks and save a NEW FAISS index to INDEX_DIR on disk (replaces the old one)."""
    embeddings = embeddings or get_embeddings(show_progress=True)
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(str(INDEX_DIR))     # writes index.faiss + index.pkl
    MODEL_FILE.write_text(EMBED_MODEL)      # so app.py can check it uses the same model
    return vectordb


def rebuild_all(embeddings=None):
    """Rebuild the whole vector DB from everything in data/. Returns (num_docs, num_chunks)."""
    docs = load_documents()
    if not docs:
        return 0, 0
    chunks = split_documents(docs)
    build_index(chunks, embeddings)
    return len(docs), len(chunks)


def add_files(file_paths, embeddings=None):
    """Add new files to the EXISTING vector DB (or create it). Returns number of chunks added."""
    docs = []
    for path in map(Path, file_paths):
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    chunks = split_documents(add_metadata(docs))
    if not chunks:
        return 0

    embeddings = embeddings or get_embeddings()
    if (INDEX_DIR / "index.faiss").exists():
        problem = index_problem()
        if problem:                      # built with another embedding model -> must rebuild
            raise RuntimeError(problem)
        vectordb = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(chunks)
        vectordb.save_local(str(INDEX_DIR))
    else:
        build_index(chunks, embeddings)
    return len(chunks)


def main():
    print(f"Reading files from '{DATA_DIR}' ...")
    print(f"Embedding with Hugging Face model '{EMBED_MODEL}' (first run downloads the model)...")
    try:
        num_docs, num_chunks = rebuild_all()
    except RuntimeError as e:
        raise SystemExit(f"\nERROR: {e}")
    if not num_docs:
        print(f"No .pdf/.txt/.md files found in '{DATA_DIR}'. Add your legal files and run again.")
        return
    print(f"\nDone. {num_docs} documents/pages -> {num_chunks} chunks.")
    print("FAISS vector DB saved locally at:")
    for f in sorted(INDEX_DIR.iterdir()):
        print(f"   {f}   ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()