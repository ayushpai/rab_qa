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

# Comment sections to extract and the label used in the annotation string
_COMMENT_SECTIONS = [
    ("FUNCTION",            "Function"),
    ("CATALYTIC ACTIVITY",  "Catalytic activity"),
    ("COFACTOR",            "Cofactor"),
    ("SUBUNIT",             "Subunit"),
    ("SUBCELLULAR LOCATION","Location"),
    ("TISSUE SPECIFICITY",  "Tissue"),
    ("DISEASE",             "Disease"),
    ("PATHWAY",             "Pathway"),
    ("PTM",                 "PTM"),
]

# Feature types that add useful annotation text
_USEFUL_FEATURES = {"DOMAIN", "REGION", "MOTIF", "REPEAT"}


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
        if piece.startswith(("Flags:", "Includes:", "Contains:")):
            continue
        for prefix in _NAME_PREFIXES:
            if piece.startswith(prefix):
                piece = piece[len(prefix):]
                break
        if piece:
            parts.append(piece)
    seen: set[str] = set()
    deduped = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return "; ".join(deduped)


_JUNK = re.compile(r'(?:Xref|Note|Evidence|IsoId|Sequence)=[^;]*;?\s*')


def _clean_section(text: str) -> str:
    """Strip curly braces, Xref/Evidence junk, and whitespace."""
    text = _strip_curly(text)
    # Remove Xref=...; Evidence=...; etc. common in CATALYTIC ACTIVITY / COFACTOR
    text = _JUNK.sub("", text)
    # Collapse whitespace
    return " ".join(text.split())


def _extract_comment_sections(record) -> dict[str, str]:
    """Extract named sections from the CC (comment) block.

    Only the FIRST occurrence of each section header is kept — UniProt
    sometimes repeats FUNCTION for isoforms/infection contexts and the
    main function always comes first.
    """
    raw = record.annotations.get("comment", "")
    if not raw:
        return {}

    # BioPython joins CC lines into one string; section headers appear at the
    # start of lines in ALL-CAPS followed by a colon.
    chunks = re.split(r'\n(?=[A-Z][A-Z /]+:)', raw)
    sections: dict[str, str] = {}

    for chunk in chunks:
        chunk = chunk.strip()
        for header, _ in _COMMENT_SECTIONS:
            if chunk.startswith(header + ":"):
                if header in sections:   # keep only first occurrence
                    break
                body = _clean_section(chunk[len(header) + 1:])
                # cap at 500 chars, break at word boundary
                if len(body) > 500:
                    body = body[:500].rsplit(" ", 1)[0]
                if body:
                    sections[header] = body
                break
    return sections


def _extract_features(record) -> list[str]:
    """Collect domain/region names from sequence features."""
    names = []
    for feat in record.features:
        if feat.type not in _USEFUL_FEATURES:
            continue
        note = feat.qualifiers.get("note", "")
        if note and len(note) < 80:
            names.append(note)
    # deduplicate while preserving order
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:8]


def _build_annotation(name: str, organism: str, length: int,
                       sections: dict[str, str], keywords: list[str],
                       features: list[str]) -> str:
    parts = [name, organism, f"{length} aa"]

    for header, label in _COMMENT_SECTIONS:
        val = sections.get(header, "")
        if val:
            parts.append(f"{label}: {val}")

    if keywords:
        parts.append("Keywords: " + "; ".join(keywords[:15]))

    if features:
        parts.append("Domains: " + "; ".join(features))

    return " | ".join(p for p in parts if p)


def parse_uniprot(
    filepath: str,
    max_records: int | None = None,
    seed: int = 42,
    force_include: list[str] | None = None,
) -> list[dict]:
    force_set = set(force_include or [])
    proteins: list[dict] = []
    opener = gzip.open if str(filepath).endswith(".gz") else open

    with opener(filepath, "rt") as handle:
        for record in tqdm(SeqIO.parse(handle, "swiss"), desc="Parsing Swiss-Prot"):
            name = _clean_name(record.description)
            organism = record.annotations.get("organism", "")
            sequence = str(record.seq)
            length = len(sequence)
            sections = _extract_comment_sections(record)
            keywords = record.annotations.get("keywords", [])
            features = _extract_features(record)
            annotation = _build_annotation(name, organism, length, sections, keywords, features)

            entry = {
                "id": record.id,
                "name": name,
                "sequence": sequence,
                "organism": organism,
                "function": sections.get("FUNCTION", ""),
                "annotation": annotation,
            }
            proteins.append(entry)

    forced = [p for p in proteins if p["id"] in force_set]
    rest = [p for p in proteins if p["id"] not in force_set]

    if max_records is not None and len(rest) > max_records:
        rng = random.Random(seed)
        rest = rng.sample(rest, max_records)

    return rest + forced


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse UniProt Swiss-Prot to JSON")
    parser.add_argument("--input", default="data/raw/uniprot_sprot.dat.gz")
    parser.add_argument("--output", default="data/processed/proteins.json")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-include", nargs="*", default=None,
        help="UniProt accessions that must be included regardless of random sampling",
    )
    args = parser.parse_args()

    proteins = parse_uniprot(
        args.input,
        max_records=args.max_records,
        seed=args.seed,
        force_include=args.force_include,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(proteins, f)

    print(f"Saved {len(proteins)} proteins to {out_path}")


if __name__ == "__main__":
    main()
