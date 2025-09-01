# 🤖  Local AI Assistant  
*RAG-powered AI for Malcolm Network Traffic Analysis (Zeek / Suricata / Arkime)*

This project builds a **local, privacy-preserving AI assistant** to query and analyze **Malcolm network telemetry**.  
It uses:
- **OpenSearch** (Malcolm’s datastore),
- **FAISS** (vector search engine),
- **SentenceTransformers** (for embeddings),
- **Ollama** (running Mistral or other LLM locally),
- **FastAPI** (backend API),
- **Streamlit** (optional simple UI).

All runs **locally** — no cloud, no external API calls.

---

## 📖 Project Overview

### 🔹 Step 1 — Data Collection
Malcolm ingests traffic into **OpenSearch** (logs from Suricata, Zeek, Arkime).

### 🔹 Step 2 — Vector Index
We extract documents, flatten them into text (`doc_to_text`), and embed them into vectors using `sentence-transformers`.  
They are stored in **FAISS** (`vectorstore.py`).

### 🔹 Step 3 — Retrieval + LLM
When you ask a question:
1. Your query is embedded → FAISS retrieves top-k relevant logs.
2. Some recent logs are fetched directly from OpenSearch (time filter).
3. Both are formatted into a **prompt**.
4. Prompt is passed to **Ollama LLM (Mistral)**.
5. The LLM returns a **SOC-style answer** (Constat → Analyse → Impact → Recommandations).

### 🔹 Step 4 — Visualization
Optionally, a simple chart of top talkers (source IPs) is created and shown in the UI.

---

## 📂 Repository Structure


| File | Purpose |
|------|---------|
| **`build_rag_index.py`** | Script to build/update the FAISS index with recent logs from OpenSearch. You run this before asking questions. |
| **`vectorstore.py`** | Lightweight wrapper around FAISS. Handles adding, upserting, deleting, and searching embeddings + stores payloads. |
| **`app.py`** | FastAPI backend. Exposes `/ask` for questions, `/healthz` for health checks, `/docs` for API docs. Queries FAISS + OpenSearch + Ollama. |
| **`ui.py`** | Streamlit UI. Simple chat-like interface for asking questions, displaying answers, sources, and a bar chart. |
| **`.env`** | Configuration file (OpenSearch URL, credentials, Ollama model, paths for FAISS, etc). **This is where you customize your environment.** |
| **`requirements.txt`** | Python dependencies. You may need to adjust versions depending on your GPU/CPU and CUDA/PyTorch availability. |

---

## ⚙️ Configuration

Before first run, edit **`.env`**:

```ini
# OpenSearch (Malcolm)
OPENSEARCH_URL=https://localhost:9200   # change to your Malcolm instance
OPENSEARCH_USER=admin
OPENSEARCH_PASS=changeme
VERIFY_SSL=false                           # use true if you have valid certs

# FAISS storage paths
VECTOR_INDEX_PATH=./data/vectors.faiss
VECTOR_PAYLOAD_PATH=./data/payload.pkl

# Embeddings model (choose a supported one)
EMBEDDINGS_MODEL=BAAI/bge-m3               # multilingual, large but slower
# Alternative (lighter, faster):
# EMBEDDINGS_MODEL=intfloat/multilingual-e5-base

# Ollama LLM
OLLAMA_BASE=http://localhost:11434         # where Ollama runs
OLLAMA_MODEL=mistral                       # choose model pulled by Ollama
