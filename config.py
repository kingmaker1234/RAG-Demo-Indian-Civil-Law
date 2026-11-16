"""
Shared settings for ingest.py and app.py
  - where the FAISS vector DB is saved on your disk
  - Hugging Face embedding model (free, runs locally)
  - LangSmith tracing (automatic once the env variables are set)
"""
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Project folder = the folder this file is in. Paths below are fixed relative to it,
# so the vector DB is always saved in the same place no matter where you run the script from.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # also loads LANGSMITH_* variables -> LangChain traces automatically

DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "faiss_index")   # <-- your local FAISS vector DB
MODEL_FILE = INDEX_DIR / "embed_model.txt"                      # remembers which model built the index

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")
TOP_K = int(os.getenv("TOP_K", "5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))

# BGE English models retrieve better when the QUESTION (not the documents) has this instruction
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


def get_embeddings(show_progress=False):
    """Hugging Face sentence-transformers model. Downloaded once, then runs offline on CPU/GPU."""
    from langchain_huggingface import HuggingFaceEmbeddings

    if EMBED_MODEL.startswith("text-embedding-"):   # common mistake: old OpenAI value left in .env
        raise RuntimeError(f"EMBED_MODEL='{EMBED_MODEL}' is an OpenAI model name, not a Hugging Face model.\n"
                           "Open .env and set:  EMBED_MODEL=BAAI/bge-small-en-v1.5")

    doc_kwargs = {"normalize_embeddings": True, "batch_size": 32}
    query_kwargs = {"normalize_embeddings": True}
    if "bge" in EMBED_MODEL.lower() and "-en" in EMBED_MODEL.lower():
        query_kwargs["prompt"] = BGE_QUERY_PROMPT

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs=doc_kwargs,
        query_encode_kwargs=query_kwargs,
        show_progress=show_progress,
    )


def index_problem():
    """Return an error message if the vector DB is missing or built with another model, else None."""
    if not (INDEX_DIR / "index.faiss").exists():
        return (f"No vector DB found at '{INDEX_DIR}'. Run 'python ingest.py' "
                "(or upload files in the chatbot sidebar).")
    if not MODEL_FILE.exists():   # index made by the first (OpenAI-embeddings) version
        return (f"The vector DB in '{INDEX_DIR}' was built by an older version with different embeddings. "
                "Delete that folder and run 'python ingest.py'.")
    if MODEL_FILE.read_text().strip() != EMBED_MODEL:
        return (f"Vector DB was built with '{MODEL_FILE.read_text().strip()}' but EMBED_MODEL is now "
                f"'{EMBED_MODEL}'. Run 'python ingest.py' again.")
    return None


def langsmith_enabled():
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def langsmith_status():
    if langsmith_enabled():
        return f"LangSmith tracing ON  -> project '{os.getenv('LANGSMITH_PROJECT', 'default')}'"
    return "LangSmith tracing OFF (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env)"
