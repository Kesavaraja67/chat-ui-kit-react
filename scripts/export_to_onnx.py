"""
Phase 1 — ONNX Export
Exports models/legal-minilm → ONNX format for NVIDIA Triton Inference Server.
Output: triton/model_repository/legal_embedding/1/model.onnx
"""
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "legal-minilm"
ONNX_OUT = BASE_DIR / "triton" / "model_repository" / "legal_embedding" / "1" / "model.onnx"
ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MeanPoolingModel(torch.nn.Module):
    """Wraps HF model + mean-pooling into a single ONNX-exportable module."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        # L2 normalize
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def main():
    print(f"📥 Loading model from {MODEL_PATH} …")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    hf_model = AutoModel.from_pretrained(str(MODEL_PATH)).to(DEVICE)
    hf_model.eval()

    wrapped = MeanPoolingModel(hf_model).to(DEVICE)
    wrapped.eval()

    # Dummy input for tracing
    sample = tokenizer(
        ["This is a legal document about tenant rights."],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = sample["input_ids"].to(DEVICE)
    attention_mask = sample["attention_mask"].to(DEVICE)

    print(f"🔄 Exporting to ONNX → {ONNX_OUT} …")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (input_ids, attention_mask),
            str(ONNX_OUT),
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    # Quick validation
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = sess.run(None, {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    })
    print(f"✅ ONNX validated — output shape: {out[0].shape}")  # (1, 384)
    print(f"✅ Saved → {ONNX_OUT}")


if __name__ == "__main__":
    main()
