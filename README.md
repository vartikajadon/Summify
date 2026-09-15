# Summify — AI Document Structuring & Trustworthy RAG Pipeline

**Summify** is an intelligent document processing engine that ingests messy, multi-file technical notes (`.md`, `.txt`, `.docx`, `.pdf`), normalizes and deduplicates content, organizes it into a standard taxonomy, exports styled documents, and provides trustworthy RAG Q&A with strict fact verification and source citations.

---

## Key Features

- **Multi-Format Ingestion (`pipeline/ingest.py`)**: Accepts `.md`, `.txt`, `.docx`, and text-extractable `.pdf` files with heading structure recovery and fault-tolerant error isolation.
- **Boundary-Aware Chunking (`pipeline/chunk.py`)**: Prefers natural boundaries (headings, paragraph breaks) over fixed windows, tracking active heading paths (`"Heading Level 1 > Subheading"`).
- **Zero-shot Taxonomy Classification (`pipeline/classify.py`)**: Classifies chunks into standard categories (`concept`, `setup/install`, `config`, `api-reference`, `example`, `troubleshooting`, `faq`, `other`) using structured LLM JSON outputs.
- **Deduplication & Multi-Format Export (`pipeline/structure.py`)**: Merges near-duplicate content across files, generates Table of Contents, appends source traceability footers (FR-3.5), and exports to Markdown (`.md`), Word (`.docx`), and PDF (`.pdf`).
- **Session-Isolated Vector Index (`pipeline/index.py`)**: Scopes ChromaDB vector collections to individual sessions (`session_<session_id>`), guaranteeing zero cross-session data leakage (FR-4.2).
- **Conversational RAG Q&A (`pipeline/chat.py`)**: Grounded answer generation with multi-turn conversation history and explicit citations (`[Source: filename > Heading]`).
- **Faithfulness & Refusal Checker (`pipeline/faithfulness.py`)**: Audits answers for factual hallucinations. If ungrounded, triggers a strict retry or outputs verbatim refusal: `"This isn't covered in the material you provided."` (FR-5.4).
- **Session Data Lifecycle (`session/manager.py`)**: One-click complete deletion removing disk directories and vector database collections (FR-6.1).
- **Streamlit Web Interface (`app/main.py`)**: Modern Stripe/Linear style minimalist light web UI with live progress spinners and interactive chat panel.

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 2. Installation
Clone the repository and install pinned dependencies:
```bash
git clone https://github.com/vartikajadon/Summify.git
cd Summify
pip install -r requirements.txt
```

---

## Configuration (`config.yaml`)

Edit `config.yaml` to switch LLM providers or customize chunking thresholds:

```yaml
# Provider Choice: 'groq' or 'ollama'
provider: groq

# Active Model Name
model: openai/gpt-oss-20b

# Max generation tokens
max_tokens: 1024

# Chunking defaults
chunking:
  max_chunk_tokens: 500
  overlap_ratio: 0.15

# Thresholds
thresholds:
  dedup_similarity: 0.85
  classifier_confidence: 0.80
  faithfulness_confidence: 0.80
```

---

## Provider Options (Hosted vs Local vs Dry-Run)

### Option A: Hosted Groq (Fast Cloud Inference)
1. Get a free API key at [console.groq.com](https://console.groq.com).
2. Set your API key in your environment:
   ```powershell
   $env:GROQ_API_KEY="your_groq_api_key_here"
   ```
3. Set `provider: groq` in `config.yaml`.

### Option B: Local Ollama (100% Offline & Free)
1. Install [Ollama](https://ollama.com).
2. Pull your desired open model:
   ```bash
   ollama pull qwen3:8b
   ```
3. Set `provider: ollama` and `model: qwen3:8b` in `config.yaml`. No API keys required!

---

## Running the Application

### 1. Launch Streamlit Web UI
```bash
streamlit run app/main.py
```
Open `http://localhost:8501` in your browser.

### 2. Run the Evaluation Harness
Run the PRD metrics benchmark suite against the fixture dataset:
```bash
python eval/metrics.py
```

### 3. Run the Automated Pytest Suite
```bash
pytest tests/ -v
```

---

## Cloud Deployment Guide

### Why Vercel Fails for Streamlit
Vercel is designed for **Serverless Functions** (FastAPI, Flask, Next.js) that export a `handler` or `app` object. **Streamlit** requires a persistent Python process with active WebSocket connections (`streamlit run app/main.py`), which Vercel Serverless Functions do not support.

### Recommended Free Deployment Options

#### Option 1: Streamlit Community Cloud (Recommended — Free & 1-Click)
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **New app** and select repository: `vartikajadon/Summify`.
3. Set **Main file path** to: `app/main.py`.
4. Under **Advanced settings -> Secrets**, add your API key:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy!**

#### Option 2: Render.com (Free Web Service)
1. Create a free account at [render.com](https://render.com).
2. Create a **New Web Service** connected to `vartikajadon/Summify`.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0`
5. Add Environment Variable: `GROQ_API_KEY`.

---

## Docker Deployment

Build and run Summify in a reproducible container:
```bash
# Build Docker image
docker build -t summify:latest .

# Run container
docker run -p 8501:8501 --env GROQ_API_KEY=$GROQ_API_KEY summify:latest
```

---

## Architecture Overview

```
summify/
├── app/
│   └── main.py              # Streamlit Web UI Application (Step 10)
├── pipeline/
│   ├── ingest.py            # Multi-format File Ingestion (Step 2)
│   ├── chunk.py             # Boundary-aware Text Chunking (Step 3)
│   ├── classify.py          # Zero-shot LLM Classifier (Step 4a)
│   ├── structure.py         # Document Assembly & Exporters (Step 5)
│   ├── index.py             # Session-Scoped Chroma Index (Step 6)
│   ├── chat.py              # Conversational RAG Engine (Step 7)
│   └── faithfulness.py      # Groundedness & Refusal Checker (Step 8a)
├── graph/
│   └── build_graph.py       # LangGraph Sequential Workflow Wiring (Step 1)
├── session/
│   └── manager.py           # Session & Data Cleanup Manager (Step 9)
├── eval/
│   └── metrics.py           # PRD Evaluation Harness (Step 11)
├── llm/
│   └── client.py            # Shared LLM Client Wrapper with Caching & Retry (Step 1)
├── tests/                   # Pytest Unit & Integration Test Suite
├── config.yaml              # Centralized Configuration File
├── Dockerfile               # Production Container Definition
├── requirements.txt         # Pinned Dependencies
└── README.md                # Technical Documentation
```

---

## License

MIT License

