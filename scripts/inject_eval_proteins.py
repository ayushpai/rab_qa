"""Fetch the 18 eval proteins from UniProt REST API and inject them into the index.

Usage:
    python -m scripts.inject_eval_proteins
"""

import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

EVAL_IDS = [
    "P04637",  # TP53
    "P00533",  # EGFR
    "P68871",  # HBB
    "P69905",  # HBA1
    "P00441",  # SOD1
    "P01308",  # INS
    "P60709",  # ACTB
    "P04406",  # GAPDH
    "P00338",  # LDHA
    "P02144",  # MB
    "P01375",  # TNF
    "P02649",  # APOE
    "P00734",  # F2 / prothrombin
    "P0A6F5",  # GroEL
    "P0A7Z4",  # RecA
    "P02787",  # TF / transferrin
    "P00390",  # GSR
    "P38398",  # BRCA1
]

PROTEINS_PATH = Path("data/processed/proteins.json")
EMB_DIR = Path("data/embeddings")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROTEIN_MODEL = "facebook/esm2_t6_8M_UR50D"
MAX_RESIDUES = 1022
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{acc}.json"


# ── UniProt fetch ─────────────────────────────────────────────────────────────

_CURLY = re.compile(r"\{[^{}]*\}")


def _strip_curly(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _CURLY.sub("", text)
    return " ".join(text.split())


def _fetch_one(acc: str) -> dict | None:
    url = UNIPROT_API.format(acc=acc)
    try:
        r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: could not fetch {acc}: {e}")
        return None
    data = r.json()

    # protein name
    try:
        rec = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except (KeyError, TypeError):
        try:
            rec = data["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
        except (KeyError, IndexError, TypeError):
            rec = acc

    # EC numbers
    try:
        ecs = [
            ec["value"]
            for ec in data["proteinDescription"]["recommendedName"].get("ecNumbers", [])
        ]
        if ecs:
            rec += "; EC=" + "; EC=".join(ecs)
    except (KeyError, TypeError):
        pass

    # organism
    try:
        organism = data["organism"]["scientificName"]
    except (KeyError, TypeError):
        organism = ""

    # sequence
    try:
        sequence = data["sequence"]["value"]
    except (KeyError, TypeError):
        sequence = ""

    # function comment
    function = ""
    for comment in data.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            for text_block in comment.get("texts", []):
                function = _strip_curly(text_block.get("value", ""))
                break
        if function:
            break

    annotation = f"{rec} | {organism} | {function}"

    return {
        "id": acc,
        "name": rec,
        "sequence": sequence,
        "organism": organism,
        "function": function,
        "annotation": annotation,
    }


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1)


def embed_text_batch(proteins: list[dict]) -> np.ndarray:
    model = SentenceTransformer(TEXT_MODEL)
    texts = [p["annotation"] for p in proteins]
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def embed_protein_batch(proteins: list[dict]) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(PROTEIN_MODEL)
    model = AutoModel.from_pretrained(PROTEIN_MODEL).to(device).eval()
    seqs = [p["sequence"][:MAX_RESIDUES] for p in proteins]
    out = []
    for seq in tqdm(seqs, desc="ESM-2"):
        inputs = tokenizer([seq], padding=True, truncation=True,
                           max_length=MAX_RESIDUES + 2, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**inputs).last_hidden_state
        pooled = _mean_pool(h, inputs["attention_mask"]).cpu().numpy().astype(np.float32)
        out.append(pooled[0])
    return np.stack(out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(PROTEINS_PATH) as f:
        proteins = json.load(f)

    existing_ids = {p["id"] for p in proteins}
    missing = [acc for acc in EVAL_IDS if acc not in existing_ids]
    if not missing:
        print("All eval proteins already in index. Nothing to do.")
        return

    print(f"Fetching {len(missing)} proteins from UniProt...")
    new_proteins = []
    for acc in tqdm(missing):
        entry = _fetch_one(acc)
        if entry:
            new_proteins.append(entry)
            print(f"  OK {acc}  {entry['name'][:60]}")
        time.sleep(0.25)  # be polite to UniProt API

    if not new_proteins:
        print("ERROR: no proteins fetched.")
        return

    print(f"\nEmbedding {len(new_proteins)} new proteins (text)...")
    new_text_emb = embed_text_batch(new_proteins)

    print(f"\nEmbedding {len(new_proteins)} new proteins (ESM-2)...")
    new_prot_emb = embed_protein_batch(new_proteins)

    # Load existing embeddings
    text_emb = np.load(EMB_DIR / "text_embeddings.npy")
    prot_emb = np.load(EMB_DIR / "protein_embeddings.npy")

    # Extend
    text_emb = np.concatenate([text_emb, new_text_emb], axis=0)
    prot_emb = np.concatenate([prot_emb, new_prot_emb], axis=0)
    proteins = proteins + new_proteins

    assert len(proteins) == text_emb.shape[0] == prot_emb.shape[0]

    # Save updated proteins list and embeddings
    with open(PROTEINS_PATH, "w") as f:
        json.dump(proteins, f)
    np.save(EMB_DIR / "text_embeddings.npy", text_emb)
    np.save(EMB_DIR / "protein_embeddings.npy", prot_emb)

    # Rebuild FAISS indices
    print("\nRebuilding FAISS indices...")
    t_emb = np.ascontiguousarray(text_emb, dtype=np.float32)
    p_emb = np.ascontiguousarray(prot_emb, dtype=np.float32)
    faiss.normalize_L2(t_emb)
    faiss.normalize_L2(p_emb)

    text_index = faiss.IndexFlatIP(t_emb.shape[1])
    text_index.add(t_emb)
    prot_index = faiss.IndexFlatIP(p_emb.shape[1])
    prot_index.add(p_emb)

    faiss.write_index(text_index, str(EMB_DIR / "text_index.faiss"))
    faiss.write_index(prot_index, str(EMB_DIR / "protein_index.faiss"))

    id_map = [p["id"] for p in proteins]
    with open(EMB_DIR / "id_map.json", "w") as f:
        json.dump(id_map, f)

    print(f"\nDone. Index now has {len(proteins)} proteins.")
    print(f"  text_emb:   {t_emb.shape}")
    print(f"  prot_emb:   {p_emb.shape}")


if __name__ == "__main__":
    main()
