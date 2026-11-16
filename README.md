# ⚖️ Indian Civil Law RAG (LangChain + FAISS + BM25 + Hugging Face + OpenAI + LangSmith)

Ask **use-case questions** ("Tenant is not vacating", "Seller took advance but won't register the sale deed")
and get answers grounded in your local legal documents — civil law (CPC, Specific Relief, Limitation,
TPA, Succession, Family, Consumer, RERA) **plus the related criminal sections with IPC → BNS mapping**.

```
data/ (PDF/TXT/MD) ──► ingest.py ──► Hugging Face embeddings (local) ──► faiss_index/  (on your disk)
                                                                              │
question ──► BM25 (keyword)  ─┐                                               │
         └─► FAISS (semantic) ┴─► EnsembleRetriever (merge) ──► OpenAI LLM ──► structured answer
                                         │                         │
                                         └────── LangSmith trace ──┘  (every question tracked)
```

| Part | Tool | Cost |
|---|---|---|
| Embeddings | Hugging Face `BAAI/bge-small-en-v1.5` via `sentence-transformers` — runs on your PC | Free |
| Vector DB | FAISS, saved in the `faiss_index/` folder | Free |
| Keyword search | BM25 (`rank-bm25`) built from the same chunks | Free |
| Answer | OpenAI chat model (`gpt-5.4-mini`) | Paid API |
| Tracking | LangSmith | Free tier available |

---

## 1. Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Linux/Mac: source venv/bin/activate

# (Optional, no NVIDIA GPU) install the smaller CPU-only PyTorch first:
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
# open .env -> paste OPENAI_API_KEY and LANGSMITH_API_KEY
```

LangSmith key: sign in at https://smith.langchain.com → **Settings → API Keys → Create API Key**.

## 2. Build the vector DB

```bash
python ingest.py
```
The first run downloads the Hugging Face model once (~130 MB); after that it works offline.
Re-run this every time you add files to `data/` or change `EMBED_MODEL`.

## 3. Start the chatbot

```bash
streamlit run streamlit_app.py   # web chatbot -> opens http://localhost:8501
python app.py                    # or: terminal chat ('new' = new chat, 'exit' = quit)
```

**Chatbot features**
- 💬 Chat with memory: ask follow-ups like *"what is the time limit for that?"* — the question is rewritten
  using the chat history before searching the vector DB.
- ⚡ Answers stream word by word; a status line shows how many chunks were found.
- 📚 Every answer has a **Sources** box (file name, PDF page, text of each chunk used).
- 🧪 Example questions to click on an empty chat.
- 📄 **Upload PDFs/TXT/MD from the sidebar** → saved to `data/uploads/` and added to the FAISS DB instantly
  (no full rebuild). "Rebuild whole DB" button re-embeds everything in `data/`.
- 🔎 Sliders for chunks per retriever and BM25 vs FAISS weight.
- 👍 / 👎 under each answer → saved as feedback on that answer's LangSmith trace.
- ⬇️ Download the conversation as Markdown; ➕ New chat.

Theme colours are in `.streamlit/config.toml`.

## 4. Track in LangSmith

Open https://smith.langchain.com → **Tracing projects → `indian-civil-law-rag`**.
Each question appears as one trace named **"Indian Civil Law RAG"** containing:
- **Rewrite follow-up question** → (only for follow-ups) the standalone search query
- **Hybrid retrieval (BM25 + FAISS)** → the BM25 and FAISS results and the merged chunks
- **Answer generation** → the exact prompt, the answer, token usage, cost and latency
- **Feedback** → the user's 👍 / 👎 from the chatbot (key `user-thumbs`, 1 = good, 0 = bad)

This shows you *why* an answer was good or bad (wrong chunks retrieved vs. LLM mistake).
To switch tracking off, set `LANGSMITH_TRACING=false` in `.env`.

---

## Where is the vector DB saved?

In the **`faiss_index/` folder inside the project folder** (always the same place, wherever you run the script from):

```
indian-civil-law-rag/
└── faiss_index/
    ├── index.faiss        # the vectors (embeddings) - the FAISS index
    ├── index.pkl          # the text chunks + metadata (file name, category) - also used to build BM25
    └── embed_model.txt    # which embedding model built this index
```
`ingest.py` prints the full path when it finishes. To store it somewhere else, set `INDEX_DIR` in `.env`
(relative to the project, or a full path like `D:/legal_db/faiss_index`).
To rebuild from scratch, delete the folder and run `python ingest.py`.

The Hugging Face model itself is cached in your user folder:
`~/.cache/huggingface/hub` (Windows: `C:\Users\<you>\.cache\huggingface\hub`).

## Project structure

```
indian-civil-law-rag/
├── streamlit_app.py     # ⭐ Streamlit chatbot (front end)
├── app.py               # RAG core: BM25 + FAISS hybrid retrieval, chat memory, streaming, LangSmith (+ terminal chat)
├── ingest.py            # load → split → embed (Hugging Face) → save FAISS; add uploaded files
├── config.py            # settings, embedding model, paths, LangSmith status
├── .streamlit/config.toml   # chatbot theme
├── requirements.txt
├── .env.example
├── faiss_index/         # created by ingest.py  <-- your local vector DB
└── data/
    ├── starter/         # included: ready-made notes (76 chunks) to test immediately
    ├── bare_acts/       # put official Acts here (PDF)
    ├── judgments/       # put SC/HC judgments here (PDF)
    └── uploads/         # files uploaded from the chatbot are saved here
```

## Choosing a Hugging Face embedding model (`EMBED_MODEL` in `.env`)

| Model | Size | When to use |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | ~130 MB | Fast on CPU, good for English legal text |
| `BAAI/bge-base-en-v1.5` | ~440 MB | Better accuracy, slower |
| `BAAI/bge-m3` | ~2.3 GB | Multilingual — ask in Hindi, Tamil, etc. |

After changing the model, run `python ingest.py` again (the app refuses to start with a mismatched index).

## Adding the full data

Drop files into `data/bare_acts/` and `data/judgments/` (any sub-folder works), then run `python ingest.py`.

| What | Where to get it |
|---|---|
| Bare Acts: CPC 1908, Specific Relief Act, Limitation Act, TPA, Registration Act, Contract Act, Hindu Succession Act, BNS 2023, BNSS 2023, BSA 2023, IPC 1860 | India Code – https://www.indiacode.nic.in |
| Supreme Court judgments | https://www.sci.gov.in |
| High Court & District judgments | eCourts judgments portal – https://judgments.ecourts.gov.in |

Tips:
- **Scanned PDFs** (images) have no text — OCR them first (e.g. `ocrmypdf input.pdf output.pdf`).
- Keep **one Act per file** with clear names (`cpc_1908.pdf`, `bns_2023.pdf`) — the file name is shown as the source.
- Thousands of pages take a few minutes to embed on CPU; a GPU is used automatically if available.

## Example questions

- My neighbour built a wall 2 feet inside my plot. What can I do?
- I paid ₹5 lakh advance for a house in 2023, the seller now refuses to register. Remedy and time limit?
- Someone forged my father's signature on a sale deed. Civil and criminal options? Give IPC and BNS sections.
- What is the BNS section for IPC 406 and 420?
- Friend took a hand loan and gave a cheque that bounced. What are the steps?
- Brothers are not giving my sister her share in ancestral property.
- Ex parte decree was passed against me without notice. What now?

## Tuning (in `.env`)

| Setting | Default | Meaning |
|---|---|---|
| `CHAT_MODEL` | `gpt-5.4-mini` | Any OpenAI chat model you have access to |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Hugging Face embedding model (see table above) |
| `TOP_K` | 5 | Chunks from BM25 **and** from FAISS (merged, duplicates removed) |
| `BM25_WEIGHT` | 0.5 | Raise to 0.6–0.7 if you mostly search by section numbers |

## Notes

- `faiss_index/index.pkl` is a pickle file — only load indexes you created yourself.
- Answers are **general legal information, not legal advice**. Always verify with the official Bare Act and an advocate.
