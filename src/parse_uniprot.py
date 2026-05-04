"""Parse UniProt Swiss-Prot flat file into a list of protein dicts."""

import argparse
import gzip
import json
import random
import re
from pathlib import Path

from Bio import SeqIO
from tqdm import tqdm


_CURLY = re.compile(r"\{[^{}]*\}")
_NAME_PREFIXES = (
    "RecName: Full=",
    "AltName: Full=",
    "SubName: Full=",
    "Short=",
    "RecName: ",
    "AltName: ",
)


def _strip_curly(text: str) -> str:
    if not text:
        return ""
    prev = None
    while prev != text:
        prev = text
        text = _CURLY.sub("", text)
    return " ".join(text.split())


def _clean_name(name: str) -> str:
    name = _strip_curly(name)
    parts: list[str] = []
    for piece in name.split(";"):
        piece = piece.strip().rstrip(",")
        if not piece:
            continue
        if piece.startswith("Flags:") or piece.startswith("Includes:") or piece.startswith("Contains:"):
            continue
        for prefix in _NAME_PREFIXES:
            if piece.startswith(prefix):
                piece = piece[len(prefix):]
                break
        if piece:
            parts.append(piece)
    seen = set()
    deduped = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return "; ".join(deduped)


def _extract_function(record) -> str:
    comment = record.annotations.get("comment", "")
    if not comment:
        return ""
    body = ""
    for section in comment.split("-!-"):
        section = section.strip()
        if section.startswith("FUNCTION:"):
            body = section[len("FUNCTION:"):].strip()
            break
    if not body and "FUNCTION:" in comment:
        idx = comment.find("FUNCTION:") + len("FUNCTION:")
        rest = comment[idx:]
        end = rest.find("-!-")
        body = rest if end == -1 else rest[:end]
    return _strip_curly(body)


def parse_uniprot(filepath: str, max_records: int | None = None, seed: int = 42) -> list[dict]:
    proteins: list[dict] = []
    opener = gzip.open if str(filepath).endswith(".gz") else open
    with opener(filepath, "rt") as handle:
        for record in tqdm(SeqIO.parse(handle, "swiss"), desc="Parsing Swiss-Prot"):
            organism = record.annotations.get("organism", "")
            function = _extract_function(record)
            name = _clean_name(record.description)
            entry = {
                "id": record.id,
                "name": name,
                "sequence": str(record.seq),
                "organism": organism,
                "function": function,
                "annotation": f"{name} | {organism} | {function}",
            }
            proteins.append(entry)

    if max_records is not None and len(proteins) > max_records:
        rng = random.Random(seed)
        proteins = rng.sample(proteins, max_records)

    return proteins


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse UniProt Swiss-Prot to JSON")
    parser.add_argument("--input", default="data/raw/uniprot_sprot.dat.gz")
    parser.add_argument("--output", default="data/processed/proteins.json")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proteins = parse_uniprot(args.input, max_records=args.max_records, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(proteins, f)

    print(f"Saved {len(proteins)} proteins to {out_path}")


if __name__ == "__main__":
    main()
