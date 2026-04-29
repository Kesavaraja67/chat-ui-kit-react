"""
Phase 1 — Embedding Model Fine-Tuning
Fine-tunes sentence-transformers/all-MiniLM-L6-v2 on the legal corpus.
Compatible with sentence-transformers >= 3.0 (uses SentenceTransformerTrainer API).
Saves fine-tuned model → models/legal-minilm/
"""
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "legal_corpus.jsonl"
MODEL_OUT = str(BASE_DIR / "models" / "legal-minilm")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32          # safe for RTX 3060 12GB
EPOCHS = 1
MAX_TRAIN_PAIRS = 5_000


def load_pairs(filepath: Path, max_pairs: int):
    """
    Build (anchor, positive) pairs from the legal corpus JSONL.
    Strategy: split each document at first newline → (title/heading, body).
    Returns a HuggingFace Dataset with columns: anchor, positive.
    """
    anchors, positives = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if len(anchors) >= max_pairs:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "")
            parts = text.split("\n", 1)
            if len(parts) == 2 and len(parts[0]) > 20 and len(parts[1]) > 50:
                anchors.append(parts[0].strip())
                positives.append(parts[1].strip()[:512])   # cap length

    # Shuffle consistently
    combined = list(zip(anchors, positives))
    random.shuffle(combined)
    anchors, positives = zip(*combined) if combined else ([], [])

    print(f"   {len(anchors):,} training pairs built")
    return Dataset.from_dict({"anchor": list(anchors), "positive": list(positives)})


def main():
    print(f"🔧 Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    if not DATA_FILE.exists():
        print(f"❌ Corpus not found at {DATA_FILE}")
        print("   Run scripts/download_dataset.py first.")
        return

    print(f"📄 Building training pairs from {DATA_FILE} …")
    train_dataset = load_pairs(DATA_FILE, MAX_TRAIN_PAIRS)

    # MultipleNegativesRankingLoss — in-batch negatives, no explicit negatives needed
    loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=MODEL_OUT,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        fp16=True,                  # AMP on RTX 3060
        bf16=False,
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_steps=100,
        evaluation_strategy="no",   # skip eval to keep it fast
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="none",           # disable wandb/tensorboard
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print(f"\n🚀 Fine-tuning for {EPOCHS} epochs …")
    trainer.train()

    print(f"\n💾 Saving fine-tuned model → {MODEL_OUT}")
    model.save_pretrained(MODEL_OUT)
    print(f"✅ Done! Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
