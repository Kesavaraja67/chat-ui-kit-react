"""
Backend unit tests — runs without GPU (mocks Triton and FAISS).
Run: pytest backend/tests/ -v
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# ── Mock embedder ─────────────────────────────────────────────────────────────
class MockEmbedder:
    def embed(self, texts, **kwargs):
        return np.random.rand(len(texts), 384).astype(np.float32)

    def embed_batched(self, texts, **kwargs):
        return self.embed(texts)


# ── Test: text chunking ────────────────────────────────────────────────────────
def test_chunk_text():
    from transformers import AutoTokenizer
    # Use a lightweight tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    text = " ".join(["This is a legal sentence."] * 100)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert len(tokens) > 50, "Expected a long token sequence"


# ── Test: /health endpoint ─────────────────────────────────────────────────────
def test_health_endpoint():
    from fastapi.testclient import TestClient
    import importlib, sys

    # Patch heavy startup
    with patch("backend.main.TritonEmbedder") as mock_emb, \
         patch("backend.main.RAGPipeline") as mock_rag, \
         patch("backend.main.faiss") as mock_faiss:

        mock_emb.return_value = MockEmbedder()
        mock_rag.return_value = MagicMock()
        mock_faiss.read_index.return_value = MagicMock()

        import backend.main as app_module
        client = TestClient(app_module.app)
        # Only test route registration
        routes = [r.path for r in app_module.app.routes]
        assert "/health" in routes
        assert "/upload" in routes
        assert "/ask" in routes


# ── Test: retrieval returns empty list when index is empty ─────────────────────
def test_rag_retrieval_empty_index():
    with patch("backend.rag_pipeline.AutoModelForCausalLM") as mock_llm, \
         patch("backend.rag_pipeline.AutoTokenizer") as mock_tok:

        mock_llm.from_pretrained.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()

        # Mock FAISS index with ntotal == 0
        mock_index = MagicMock()
        mock_index.ntotal = 0

        from backend.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(
            embedder=MockEmbedder(),
            faiss_index=mock_index,
            metadata=[],
            model_path="gpt2",
        )
        results = pipeline.retrieve("What are tenant rights?")
        assert results == []


# ── Test: document chunking produces non-empty chunks ────────────────────────
def test_document_chunking_produces_chunks():
    with patch("backend.rag_pipeline.AutoModelForCausalLM") as mock_llm, \
         patch("backend.rag_pipeline.AutoTokenizer") as mock_tok:

        from unittest.mock import MagicMock
        real_tok_mock = MagicMock()
        real_tok_mock.encode.return_value = list(range(500))
        real_tok_mock.decode.side_effect = lambda tokens, **_: f"chunk_{len(tokens)}"
        mock_tok.from_pretrained.return_value = real_tok_mock
        mock_llm.from_pretrained.return_value = MagicMock()

        mock_index = MagicMock()
        mock_index.ntotal = 0

        from backend.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(
            embedder=MockEmbedder(),
            faiss_index=mock_index,
            metadata=[],
            model_path="gpt2",
        )
        chunks = pipeline._chunk_text("word " * 1000)
        assert len(chunks) > 1
