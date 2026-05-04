"""Build FAISS indices for text and protein embeddings."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def build_indices(
    text_embeddings: np.ndarray,
    protein_embeddings: np.ndarray,
    proteins: list[dict],
    out_dir: str = "data/embeddings",
) -> tuple[faiss.IndexFlatIP, faiss.IndexFlatIP]:
    text_embeddings = np.ascontiguousarray(text_embeddings, dtype=np.float32)
    protein_embeddings = np.ascontiguousarray(protein_embeddings, dtype=np.float32)
    faiss.normalize_L2(text_embeddings)
    faiss.normalize_L2(protein_embeddings)

    text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
    text_index.add(text_embeddings)

    protein_index = faiss.IndexFlatIP(protein_embeddings.shape[1])
    protein_index.add(protein_embeddings)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(text_index, str(out / "text_index.faiss"))
    faiss.write_index(protein_index, str(out / "protein_index.faiss"))

    id_map = [p["id"] for p in proteins]
    with open(out / "id_map.json", "w") as f:
        json.dump(id_map, f)

    print(f"Built text index ({text_embeddings.shape}) and protein index ({protein_embeddings.shape})")
    print(f"Saved to {out}/")
    return text_index, protein_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices")
    parser.add_argument("--proteins", default="data/processed/proteins.json")
    parser.add_argument("--text-embeddings", default="data/embeddings/text_embeddings.npy")
    parser.add_argument("--protein-embeddings", default="data/embeddings/protein_embeddings.npy")
    parser.add_argument("--out-dir", default="data/embeddings")
    args = parser.parse_args()

    with open(args.proteins) as f:
        proteins = json.load(f)
    text_emb = np.load(args.text_embeddings)
    protein_emb = np.load(args.protein_embeddings)

    assert len(proteins) == text_emb.shape[0] == protein_emb.shape[0], (
        f"Mismatched row counts: proteins={len(proteins)}, "
        f"text={text_emb.shape[0]}, protein={protein_emb.shape[0]}"
    )

    build_indices(text_emb, protein_emb, proteins, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
