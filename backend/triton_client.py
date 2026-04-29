"""
Local Embedder Client — bypasses Triton Inference Server to run locally via SentenceTransformers.
Handles tokenization and inference locally (CPU/GPU).
"""
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load the local model directly
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models" / "legal-minilm"

class TritonEmbedder:
    """
    Mock TritonEmbedder that actually runs locally via SentenceTransformer.
    This keeps the rest of the backend API identical while bypassing Docker.
    """
    def __init__(self, triton_url: str = None, tokenizer_path: str = None, model_name: str = "legal_embedding"):
        print("🔧 Loading local SentenceTransformer model instead of Triton...")
        if MODEL_DIR.exists():
            self.model = SentenceTransformer(str(MODEL_DIR))
            print("✅ Loaded fine-tuned local model")
        else:
            print("⚠️ Fine-tuned model not found, falling back to base model")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts: list[str], max_length: int = 256) -> np.ndarray:
        """Embed a list of texts locally. Returns L2-normalised numpy array."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def embed_batched(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed in batches locally."""
        embeddings = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)
