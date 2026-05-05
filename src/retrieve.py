"""Retriever: text-only, protein-only, and hybrid retrieval over FAISS indices."""

import json
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
TEXT_MODEL_NAME = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
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

        # BM25 index over annotation strings, aligned with id_map order
        corpus = [
            re.sub(r"[^a-z0-9 ]", " ", self.proteins[pid]["annotation"].lower()).split()
            for pid in self.id_map
        ]
        self.bm25 = BM25Okapi(corpus)

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

    def retrieve_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        tokens = re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
        scores = self.bm25.get_scores(tokens)
        top_idxs = np.argpartition(scores, -top_k)[-top_k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]
        return self._lookup(top_idxs, scores[top_idxs])

    def retrieve_hybrid(
        self, query: str, sequence: str, top_k: int = 5, rrf_k: int = 60,
    ) -> list[dict]:
        text_vec = self._encode_text(query)
        prot_vec = self._encode_sequence(sequence)

        pool = top_k * 10
        t_scores, t_idxs = self.text_index.search(text_vec, pool)
        p_scores, p_idxs = self.protein_index.search(prot_vec, pool)

        # BM25 exact-keyword ranking
        tokens = re.sub(r"[^a-z0-9 ]", " ", query.lower()).split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = np.argpartition(bm25_scores, -pool)[-pool:]
        bm25_top = bm25_top[np.argsort(bm25_scores[bm25_top])[::-1]]

        # RRF over all three modalities: dense-text + BM25 + protein
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(t_idxs[0]):
            if idx < 0:
                continue
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_top):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(p_idxs[0]):
            if idx < 0:
                continue
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            pid = self.id_map[idx]
            entry = dict(self.proteins[pid])
            entry["_score"] = score
            results.append(entry)
        return results

    def retrieve(
        self,
        query: str,
        sequence: str | None = None,
        mode: str = "auto",
        top_k: int = 5,
    ) -> list[dict]:
        if mode == "auto":
            if sequence:
                mode = "hybrid"
            elif self.is_sequence_query(query):
                mode = "protein"
            else:
                mode = "text"

        if mode == "text" or not sequence:
            return self.retrieve_text(query, top_k=top_k)
        if mode == "protein":
            return self.retrieve_protein(sequence, top_k=top_k)
        if mode == "hybrid":
            return self.retrieve_hybrid(query, sequence, top_k=top_k)
        raise ValueError(f"Unknown mode: {mode}")
