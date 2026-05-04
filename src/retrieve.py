"""Retriever: text-only, protein-only, and hybrid retrieval over FAISS indices."""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROTEIN_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_RESIDUES = 1022


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


class Retriever:
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        proteins_path: str = "data/processed/proteins.json",
        text_model_name: str = TEXT_MODEL_NAME,
        protein_model_name: str = PROTEIN_MODEL_NAME,
        device: str | None = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.text_index = faiss.read_index(str(self.embeddings_dir / "text_index.faiss"))
        self.protein_index = faiss.read_index(str(self.embeddings_dir / "protein_index.faiss"))

        with open(self.embeddings_dir / "id_map.json") as f:
            self.id_map: list[str] = json.load(f)

        with open(proteins_path) as f:
            proteins_list = json.load(f)
        self.proteins: dict[str, dict] = {p["id"]: p for p in proteins_list}

        self.text_model_name = text_model_name
        self.protein_model_name = protein_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._text_model: SentenceTransformer | None = None
        self._protein_tokenizer = None
        self._protein_model = None

    @property
    def text_model(self) -> SentenceTransformer:
        if self._text_model is None:
            self._text_model = SentenceTransformer(self.text_model_name)
        return self._text_model

    def _load_protein_model(self):
        if self._protein_model is None:
            self._protein_tokenizer = AutoTokenizer.from_pretrained(self.protein_model_name)
            self._protein_model = (
                AutoModel.from_pretrained(self.protein_model_name).to(self.device).eval()
            )
        return self._protein_tokenizer, self._protein_model

    @staticmethod
    def is_sequence_query(query: str) -> bool:
        stripped = "".join(query.split())
        if not stripped:
            return False
        valid = sum(1 for c in stripped if c in AMINO_ACIDS)
        return valid / len(stripped) > 0.6

    def _encode_text(self, query: str) -> np.ndarray:
        vec = self.text_model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        tokenizer, model = self._load_protein_model()
        seq = sequence[:MAX_RESIDUES]
        inputs = tokenizer(
            [seq], padding=True, truncation=True, max_length=MAX_RESIDUES + 2, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        vec = pooled.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def _lookup(self, idxs: np.ndarray, scores: np.ndarray) -> list[dict]:
        results = []
        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            pid = self.id_map[idx]
            entry = dict(self.proteins[pid])
            entry["_score"] = float(score)
            results.append(entry)
        return results

    def retrieve_text(self, query: str, top_k: int = 5) -> list[dict]:
        vec = self._encode_text(query)
        scores, idxs = self.text_index.search(vec, top_k)
        return self._lookup(idxs[0], scores[0])

    def retrieve_protein(self, sequence: str, top_k: int = 5) -> list[dict]:
        vec = self._encode_sequence(sequence)
        scores, idxs = self.protein_index.search(vec, top_k)
        return self._lookup(idxs[0], scores[0])

    def retrieve_hybrid(
        self, query: str, sequence: str, top_k: int = 5, alpha: float = 0.5
    ) -> list[dict]:
        text_vec = self._encode_text(query)
        prot_vec = self._encode_sequence(sequence)

        pool = top_k * 4
        t_scores, t_idxs = self.text_index.search(text_vec, pool)
        p_scores, p_idxs = self.protein_index.search(prot_vec, pool)

        merged: dict[int, dict] = {}
        for i, s in zip(t_idxs[0], t_scores[0]):
            if i < 0:
                continue
            merged.setdefault(int(i), {"text": 0.0, "protein": 0.0})["text"] = float(s)
        for i, s in zip(p_idxs[0], p_scores[0]):
            if i < 0:
                continue
            merged.setdefault(int(i), {"text": 0.0, "protein": 0.0})["protein"] = float(s)

        ranked = sorted(
            merged.items(),
            key=lambda kv: alpha * kv[1]["text"] + (1 - alpha) * kv[1]["protein"],
            reverse=True,
        )[:top_k]

        results = []
        for idx, parts in ranked:
            pid = self.id_map[idx]
            entry = dict(self.proteins[pid])
            entry["_score"] = alpha * parts["text"] + (1 - alpha) * parts["protein"]
            entry["_score_text"] = parts["text"]
            entry["_score_protein"] = parts["protein"]
            results.append(entry)
        return results

    def retrieve(
        self,
        query: str,
        sequence: str | None = None,
        mode: str = "auto",
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict]:
        if mode == "auto":
            if sequence:
                mode = "hybrid"
            elif self.is_sequence_query(query):
                mode = "protein"
            else:
                mode = "text"

        if mode == "text":
            return self.retrieve_text(query, top_k=top_k)
        if mode == "protein":
            seq = sequence or query.strip()
            return self.retrieve_protein(seq, top_k=top_k)
        if mode == "hybrid":
            seq = sequence or query.strip()
            return self.retrieve_hybrid(query, seq, top_k=top_k, alpha=alpha)
        raise ValueError(f"Unknown mode: {mode}")
