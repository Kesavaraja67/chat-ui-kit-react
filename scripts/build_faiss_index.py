"""
Phase 3 — FAISS Index Builder (Local PyTorch Fallback)
Chunks the legal dataset, embeds it using the local SentenceTransformer (mocking Triton),
and builds a GPU-accelerated FAISS IVF index.
"""
import json
import pickle
import time
from pathlib import Path
from typing import Generator

import faiss
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from backend.triton_client import TritonEmbedder

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "legal_corpus.jsonl"
INDEX_OUT = BASE_DIR / "data" / "faiss.index"
META_OUT = BASE_DIR / "data" / "chunk_metadata.pkl"

CHUNK_SIZE = 256        # tokens per chunk
CHUNK_OVERLAP = 32
BATCH_SIZE = 64

def iter_chunks(filepath: Path) -> Generator[tuple[str, dict], None, None]:
    """Yields (chunk_text, metadata_dict) pairs."""
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            # Simple text chunking by words to avoid heavy tokenization overhead
            words = text.split()
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = " ".join(words[i : i + CHUNK_SIZE])
                if len(chunk) > 20:
                    yield chunk, {"text": chunk, "url": rec.get("url", ""), "date": rec.get("created_timestamp", "")}

def main():
    print("🔗 Connecting to TritonEmbedder (Local mock)")
    embedder = TritonEmbedder()

    # 1. Initialize FAISS index
    dim = 384
    nlist = 100
    quantizer = faiss.IndexFlatIP(dim)
    index_cpu = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    all_embeddings = []
    all_metadata = []
    batch_texts, batch_metas = [], []

    def flush_batch():
        if not batch_texts:
            return
        embs = embedder.embed_batched(batch_texts, batch_size=BATCH_SIZE)
        all_embeddings.append(embs)
        all_metadata.extend(batch_metas)
        batch_texts.clear()
        batch_metas.clear()

    total = 0
    print("⚙️  Embedding corpus chunks …")
    
    # We limit to 50k chunks so it runs under a few minutes locally
    MAX_CHUNKS = 50_000 
    
    t0 = time.time()
    for chunk_text_str, meta in iter_chunks(DATA_FILE):
        if total >= MAX_CHUNKS:
            break
        batch_texts.append(chunk_text_str)
        batch_metas.append(meta)
        if len(batch_texts) == BATCH_SIZE:
            flush_batch()
            total += BATCH_SIZE
            if total % 5000 == 0:
                print(f"   Embedded {total:,} chunks …")

    flush_batch()

    if not all_embeddings:
        print("❌ No data to index")
        return

    embeddings_np = np.vstack(all_embeddings)
    t_embed = time.time() - t0
    print(f"✅ Embedded {embeddings_np.shape[0]:,} chunks in {t_embed:.1f}s")

    # 2. Train and Add to FAISS
    print("🧠 Training FAISS IVF index …")
    index.train(embeddings_np)
    index.add(embeddings_np)
    print(f"   FAISS ntotal: {index.ntotal}")

    # 3. Save
    print(f"💾 Saving index → {INDEX_OUT}")
    index_cpu = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index_cpu, str(INDEX_OUT))

    print(f"💾 Saving metadata → {META_OUT}")
    with open(META_OUT, "wb") as f:
        pickle.dump(all_metadata, f)

    print("🎉 Done! FAISS database is ready.")

if __name__ == "__main__":
    main()
