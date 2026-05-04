"""Embed protein sequences with ESM-2 (facebook/esm2_t6_8M_UR50D, 320-dim)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_RESIDUES = 1022  # ESM-2 hard limit (excluding BOS/EOS)
EMBED_DIM = 320


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def embed_proteins(
    proteins: list[dict],
    batch_size: int = 16,
    model_name: str = MODEL_NAME,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    n = len(proteins)
    seqs = [p["sequence"][:MAX_RESIDUES] for p in proteins]

    order = sorted(range(n), key=lambda i: len(seqs[i]))
    out = np.zeros((n, EMBED_DIM), dtype=np.float32)

    for start in tqdm(range(0, n, batch_size), desc="ESM-2 embeddings"):
        idxs = order[start : start + batch_size]
        batch_seqs = [seqs[i] for i in idxs]
        inputs = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=MAX_RESIDUES + 2,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = pooled.cpu().numpy().astype(np.float32)
        for j, i in enumerate(idxs):
            out[i] = pooled[j]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed sequences with ESM-2")
    parser.add_argument("--input", default="data/processed/proteins.json")
    parser.add_argument("--output", default="data/embeddings/protein_embeddings.npy")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.input) as f:
        proteins = json.load(f)

    embeddings = embed_proteins(proteins, batch_size=args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Saved protein embeddings of shape {embeddings.shape} to {out_path}")


if __name__ == "__main__":
    main()
