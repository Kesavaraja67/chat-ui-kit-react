# LegalMind — Document Q&A System
### RAG + TinyLlama + FastAPI + React + NVIDIA Triton

<div align="center">

![LegalMind Banner](https://img.shields.io/badge/LegalMind-RAG%20Legal%20Q%26A-c9a84c?style=for-the-badge&logo=scales)
![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-76b900?style=for-the-badge&logo=nvidia)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi)

</div>

## Overview

**LegalMind** is an end-to-end, GPU-accelerated Document Q&A system built for the legal domain. It combines:

- **Retrieval-Augmented Generation (RAG)** to ground LLM answers in uploaded documents
- **NVIDIA Triton Inference Server** to serve the embedding model with GPU batching
- **FAISS GPU** vector store for fast similarity search
- **TinyLlama-1.1B-Chat** as the open-source LLM
- **React + chatscope/chat-ui-kit-react** for a professional chat interface
- **Fine-tuned `all-MiniLM-L6-v2`** on 480MB of legal text from Pile of Law

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (React + chatscope)                                     │
│  ┌────────────┐   ┌──────────────────────────────────────────┐  │
│  │ Sidebar    │   │  ChatContainer                           │  │
│  │ Doc Upload │   │  MessageList / MessageInput              │  │
│  │ FAISS Stats│   │  Source Citation Chips                   │  │
│  └────────────┘   └──────────────────────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼──────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                                    │
│  /upload → chunk → Triton embed → FAISS add                     │
│  /ask    → Triton embed → FAISS search → TinyLlama generate     │
│  /ask/stream → SSE token-by-token streaming                     │
└────────────┬──────────────────────────┬─────────────────────────┘
             │ gRPC :8001               │ GPU (cuda:0)
┌────────────▼────────────┐  ┌──────────▼───────────────────────┐
│  NVIDIA Triton Server   │  │  TinyLlama-1.1B-Chat             │
│  legal_embedding (ONNX) │  │  HuggingFace Transformers        │
│  GPU batching, dynamic  │  │  torch.float16, device_map=auto  │
│  batching, CUDA accel.  │  │  TextIteratorStreamer             │
└─────────────────────────┘  └──────────────────────────────────┘
             │
┌────────────▼────────────┐
│  FAISS GPU IVF Index    │
│  ~384-dim embeddings    │
│  faiss-gpu-cu12         │
└─────────────────────────┘
```

---

## Hardware Requirements

| Component | Minimum | Used in this project |
|---|---|---|
| GPU | NVIDIA with CUDA ≥ 11.8 | RTX 3060 12GB |
| VRAM | 8GB | 12GB |
| RAM | 16GB | - |
| Storage | 10GB | - |
| CUDA Driver | 520+ | 580 (CUDA 13.0) |

---

## Quick Start

### 1. Prerequisites

```bash
# Check GPU
nvidia-smi

# Install nvidia-container-toolkit (for Docker GPU access)
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Run the pipeline scripts (one-time setup)

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Step 1: Download the legal corpus (~480MB)
python scripts/download_dataset.py

# Step 2: Fine-tune the embedding model (~2-4 hours)
python scripts/finetune_embeddings.py

# Step 3: Export to ONNX for Triton
python scripts/export_to_onnx.py

# Step 4: Start Triton (needed for index building)
docker compose up triton -d

# Step 5: Build the FAISS index
python scripts/build_faiss_index.py
```

### 3. Start with Docker Compose

```bash
docker compose up --build
```

| Service | URL | Purpose |
|---|---|---|
| Frontend | http://localhost:3000 | React chat UI |
| Backend | http://localhost:8000 | FastAPI RAG API |
| Triton Metrics | http://localhost:8002/metrics | GPU inference stats |

### 4. Development (without Docker)

```bash
# Terminal 1 — Frontend
cd frontend && npm run dev       # → http://localhost:3000

# Terminal 2 — Backend
cd backend && uvicorn main:app --reload --port 8000

# Terminal 3 — Triton (requires Docker)
docker compose up triton
```

---

## Project Structure

```
chat-ui-kit-react/              ← chatscope component library (source)
├── src/                        ← UI kit components (DO NOT MODIFY)
├── dist/                       ← Built library (es + cjs)
│
├── frontend/                   ← Vite React app
│   ├── src/
│   │   ├── App.jsx             ← Root layout
│   │   ├── index.css           ← Dark legal theme
│   │   ├── components/
│   │   │   ├── ChatPanel.jsx   ← chatscope message list/input
│   │   │   └── DocumentPanel.jsx  ← drag-and-drop upload
│   │   ├── hooks/
│   │   │   └── useChat.js      ← SSE streaming state hook
│   │   └── api/
│   │       └── index.js        ← FastAPI client
│   └── vite.config.js
│
├── backend/                    ← FastAPI server
│   ├── main.py                 ← /upload /ask /health /sources /clear
│   ├── rag_pipeline.py         ← FAISS retrieval + TinyLlama generation
│   ├── triton_client.py        ← gRPC Triton embedder
│   ├── requirements.txt
│   └── tests/
│       └── test_backend.py
│
├── triton/
│   └── model_repository/
│       └── legal_embedding/
│           ├── config.pbtxt    ← GPU batching config
│           └── 1/
│               └── model.onnx  ← (generated by export_to_onnx.py)
│
├── scripts/
│   ├── download_dataset.py     ← pile-of-law r_legaladvice ~480MB
│   ├── finetune_embeddings.py  ← sentence-transformers fine-tuning
│   ├── export_to_onnx.py       ← HF → ONNX export
│   ├── build_faiss_index.py    ← GPU FAISS IVF index builder
│   └── export_docker.sh        ← Docker save + RAR + Hub push
│
├── data/                       ← legal_corpus.jsonl, faiss.index, metadata.pkl
├── models/legal-minilm/        ← fine-tuned embedding model
├── docs/                       ← software paper sections
│
├── docker-compose.yml          ← 3-service GPU stack
├── Dockerfile.frontend
├── Dockerfile.backend
└── README.md
```

---

## API Reference

### `POST /upload`
Upload a PDF or TXT file for indexing.

```json
// Response
{ "doc_id": "abc12345", "name": "contract.pdf", "chunks_indexed": 42 }
```

### `POST /ask`
Ask a question against the indexed documents.

```json
// Request
{ "query": "What are the tenant's obligations under this lease?", "top_k": 5 }

// Response
{
  "answer": "Under the lease, the tenant is obligated to...",
  "sources": [
    { "text": "...", "source": "contract.pdf", "score": 0.912 }
  ],
  "query": "What are the tenant's obligations..."
}
```

### `POST /ask/stream`
Same as `/ask` but returns Server-Sent Events for token streaming.

### `GET /health`
```json
{ "status": "ok", "triton": true, "llm": true }
```

---

## Dataset

**Source:** `pile-of-law/pile-of-law` — `r_legaladvice` subset  
**Size:** ~480MB uncompressed (200K Q&A rows)  
**License:** CC-BY-NC-SA 4.0 (non-commercial research only)  
**Citation:** Henderson et al. (2022), "Pile of Law", arXiv:2207.00220

---

## Docker Export

```bash
# Build, export to RAR, and push to Docker Hub
chmod +x scripts/export_docker.sh
./scripts/export_docker.sh your-dockerhub-username
```

---

## Software Paper Reference

This system is documented as a software paper covering:
1. System Architecture (microservices design)
2. Dataset & Fine-Tuning Methodology
3. RAG Pipeline Design
4. NVIDIA Triton GPU Optimization
5. Performance Benchmarks (CPU vs GPU inference latency)
6. Scalability Analysis

See [`docs/paper_outline.md`](docs/paper_outline.md) for the full structure.

---

## License

MIT (frontend) · Apache-2.0 (NVIDIA components) · CC-BY-NC-SA 4.0 (dataset)
