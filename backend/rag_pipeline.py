"""
RAG Pipeline — Orchestrates embedding → retrieval → generation.
LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0 running on GPU via HuggingFace Transformers.
"""
import pickle
from pathlib import Path
from typing import AsyncGenerator

import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

from triton_client import TritonEmbedder

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
TINYLLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = (
    "You are LegalMind, an expert legal research assistant. "
    "Answer questions strictly based on the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Be precise, cite relevant legal concepts, and keep answers concise."
)


class RAGPipeline:
    def __init__(
        self,
        embedder: TritonEmbedder,
        faiss_index,
        metadata: list[dict],
        model_path: str,
    ):
        self.embedder = embedder
        self.index = faiss_index
        self.metadata = metadata
        self.tokenizer_path = model_path

        # Load TinyLlama tokenizer for chunking
        from transformers import AutoTokenizer as AT
        self.chunk_tokenizer = AT.from_pretrained(model_path)

        print("🤖 Loading TinyLlama-1.1B-Chat …")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_ID)
        self.llm = AutoModelForCausalLM.from_pretrained(
            TINYLLAMA_ID,
            torch_dtype=torch.float16,
            device_map="auto",           # RTX 3060 → cuda:0
        )
        self.llm.eval()
        print("✅ TinyLlama loaded on", next(self.llm.parameters()).device)

    # ── Document Ingestion ────────────────────────────────────────────────────
    def add_document(self, text: str, doc_id: str, source: str) -> int:
        """Chunk → embed via Triton → add to FAISS. Returns number of chunks."""
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        embeddings = self.embedder.embed_batched(chunks, batch_size=32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        for chunk in chunks:
            self.metadata.append({"doc_id": doc_id, "source": source, "text": chunk})

        return len(chunks)

    def clear(self):
        self.index.reset()
        self.metadata.clear()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Embed query → cosine search → return top_k chunks with scores."""
        if self.index.ntotal == 0:
            return []
        q_emb = self.embedder.embed([query])  # (1, 384)
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    # ── Generation ────────────────────────────────────────────────────────────
    def answer(self, query: str, top_k: int = 5) -> tuple[str, list[dict]]:
        sources = self.retrieve(query, top_k)
        context = self._build_context(sources)
        prompt = self._build_prompt(query, context)

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = self.llm_tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer, sources

    async def answer_stream(self, query: str, top_k: int = 5) -> AsyncGenerator[str, None]:
        """Token-by-token streaming using TextIteratorStreamer."""
        sources = self.retrieve(query, top_k)
        context = self._build_context(sources)
        prompt = self._build_prompt(query, context)

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        streamer = TextIteratorStreamer(
            self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )

        thread = threading.Thread(target=self.llm.generate, kwargs=generate_kwargs)
        thread.start()

        import asyncio
        for token in streamer:
            yield token
            await asyncio.sleep(0)

        thread.join()

    # ── Private helpers ───────────────────────────────────────────────────────
    def _chunk_text(self, text: str) -> list[str]:
        tokens = self.chunk_tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunk = self.chunk_tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            if len(chunk.strip()) > 20:
                chunks.append(chunk.strip())
            if end == len(tokens):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _build_context(self, sources: list[dict]) -> str:
        if not sources:
            return "No relevant documents found."
        parts = []
        for i, s in enumerate(sources, 1):
            parts.append(f"[Source {i} — {s.get('source', 'corpus')} (score: {s['score']:.3f})]:\n{s['text']}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\nContext:\n{context}\n\nQuestion: {query}</s>\n"
            "<|assistant|>\n"
        )
