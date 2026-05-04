"""Embed protein annotation text with sentence-transformers."""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_text(proteins: list[dict], batch_size: int = 64, model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [p["annotation"] for p in proteins]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed protein annotations as text vectors")
    parser.add_argument("--input", default="data/processed/proteins.json")
    parser.add_argument("--output", default="data/embeddings/text_embeddings.npy")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.input) as f:
        proteins = json.load(f)

    embeddings = embed_text(proteins, batch_size=args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Saved text embeddings of shape {embeddings.shape} to {out_path}")


if __name__ == "__main__":
    main()
