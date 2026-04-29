"""
FastAPI Backend — LegalMind RAG Q&A System
Endpoints: /health  /upload  /ask  /sources  /clear
"""
import io
import os
import pickle
import uuid
from pathlib import Path
from typing import AsyncGenerator

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_pipeline import RAGPipeline
from triton_client import TritonEmbedder

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
META_PATH = BASE_DIR / "data" / "chunk_metadata.pkl"
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "legal-minilm"))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LegalMind API",
    description="RAG-powered Legal Document Q&A with TinyLlama + FAISS + Triton",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialised at startup) ─────────────────────────────────────────
embedder: TritonEmbedder = None
rag: RAGPipeline = None
documents: dict[str, dict] = {}          # doc_id → {name, size, chunks}


@app.on_event("startup")
async def startup():
    global embedder, rag
    print("🚀 Connecting to Triton …")
    embedder = TritonEmbedder(triton_url=TRITON_URL, tokenizer_path=MODEL_PATH)

    print("🏗️  Loading FAISS index …")
    if INDEX_PATH.exists() and META_PATH.exists():
        index_cpu = faiss.read_index(str(INDEX_PATH))
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        print("   ⚠️  No pre-built index found — starting with empty index.")
        index_gpu = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), 384)
        metadata = []

    print("🤖 Loading TinyLlama …")
    rag = RAGPipeline(
        embedder=embedder,
        faiss_index=index_gpu,
        metadata=metadata,
        model_path=MODEL_PATH,
    )
    print("✅ LegalMind backend ready!")


# ── Schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "triton": embedder is not None, "llm": rag is not None}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Parse PDF or TXT, chunk, embed via Triton, add to FAISS index."""
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    content = await file.read()
    doc_id = str(uuid.uuid4())[:8]
    text = _extract_text(content, file.filename)
    num_chunks = rag.add_document(text, doc_id=doc_id, source=file.filename)

    documents[doc_id] = {
        "id": doc_id,
        "name": file.filename,
        "size_kb": round(len(content) / 1024, 1),
        "chunks": num_chunks,
    }
    return {"doc_id": doc_id, "name": file.filename, "chunks_indexed": num_chunks}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """RAG Q&A — embed query → FAISS search → TinyLlama answer."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    answer, sources = rag.answer(req.query, top_k=req.top_k)
    return AskResponse(answer=answer, sources=sources, query=req.query)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Streaming SSE version of /ask."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")

    async def generate() -> AsyncGenerator[str, None]:
        async for token in rag.answer_stream(req.query, top_k=req.top_k):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/sources")
async def list_sources():
    return {"documents": list(documents.values())}


@app.delete("/clear")
async def clear_index():
    global rag
    rag.clear()
    documents.clear()
    return {"status": "index cleared"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_text(content: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    return content.decode("utf-8", errors="replace")
