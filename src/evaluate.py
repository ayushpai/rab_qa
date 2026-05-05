"""Evaluate retrieval and generation quality across retrieval modes."""

import argparse
import json
import math
import re
import time
from pathlib import Path

from tqdm import tqdm

from src.generate import DEFAULT_MODEL, _call_with_retry, generate_answer, make_client
from src.retrieve import Retriever


JUDGE_PROMPT = """Rate this answer on a scale of 1-5 for each dimension:
- Correctness: does it match the ground truth?
- Groundedness: is it supported by the retrieved context?
- Completeness: does it fully answer the question?

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}

Respond in JSON: {{"correctness": X, "groundedness": X, "completeness": X}}"""


def _recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hit = sum(1 for r in relevant_ids if r in top_k)
    return hit / len(relevant_ids)


def _mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def _map(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Average Precision over the retrieved list."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    hits = 0
    precision_sum = 0.0
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant_ids)


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """NDCG@k with binary relevance."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, rid in enumerate(retrieved_ids[:k], 1)
        if rid in relevant
    )
    # ideal: all relevant docs ranked first
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _all_metrics(retrieved_ids: list[str], relevant_ids: list[str]) -> dict:
    return {
        "recall_at_1":  _recall_at_k(retrieved_ids, relevant_ids, 1),
        "recall_at_3":  _recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall_at_5":  _recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": _recall_at_k(retrieved_ids, relevant_ids, 10),
        "mrr":          _mrr(retrieved_ids, relevant_ids),
        "map":          _map(retrieved_ids, relevant_ids),
        "ndcg_at_5":    _ndcg_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg_at_10":   _ndcg_at_k(retrieved_ids, relevant_ids, 10),
    }


def _modes_for_question(q: dict, modes: tuple[str, ...]) -> list[str]:
    return list(modes)


def evaluate(
    questions_path: str,
    retriever: Retriever,
    client,
    modes: tuple[str, ...] = ("text", "protein", "hybrid"),
    top_k: int = 5,
    out_path: str = "eval/results.json",
    skip_generate: bool = False,
    model: str | None = None,
) -> dict:
    with open(questions_path) as f:
        questions = json.load(f)

    METRIC_KEYS = [
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "mrr", "map", "ndcg_at_5", "ndcg_at_10",
    ]

    results = []
    for q in tqdm(questions, desc="Evaluating"):
        per_q = {
            "id": q["id"],
            "category": q.get("category", "unknown"),
            "question": q["question"],
            "ground_truth": q.get("ground_truth", ""),
            "relevant_uniprot_ids": q.get("relevant_uniprot_ids", []),
            "modes": {},
        }
        for mode in _modes_for_question(q, modes):
            t0 = time.perf_counter()
            retrieved = retriever.retrieve(
                q["question"], sequence=q.get("sequence"), mode=mode, top_k=top_k
            )
            retrieve_time = time.perf_counter() - t0

            retrieved_ids = [p["id"] for p in retrieved]
            metrics = _all_metrics(retrieved_ids, q.get("relevant_uniprot_ids", []))

            answer = ""
            gen_time = 0.0
            if not skip_generate:
                t0 = time.perf_counter()
                kwargs = {"model": model} if model else {}
                answer = generate_answer(q["question"], retrieved, client, **kwargs)
                gen_time = time.perf_counter() - t0

            per_q["modes"][mode] = {
                "retrieved_ids": retrieved_ids,
                "retrieved_names": [p.get("name", "") for p in retrieved],
                **metrics,
                "answer": answer,
                "retrieve_time_s": retrieve_time,
                "generate_time_s": gen_time,
            }
        results.append(per_q)

    summary: dict = {"per_mode": {}}
    for mode in modes:
        mode_rows = [r["modes"][mode] for r in results if mode in r["modes"]]
        if mode_rows:
            n = len(mode_rows)
            summary["per_mode"][mode] = {
                "n": n,
                **{k: sum(row[k] for row in mode_rows) / n for k in METRIC_KEYS},
            }

    out = {"summary": summary, "results": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Retrieval summary ===")
    header = f"  {'mode':8s}  {'n':>3}  R@1    R@3    R@5    R@10   MRR    MAP    NDCG@5 NDCG@10"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode, s in summary["per_mode"].items():
        print(
            f"  {mode:8s}  {s['n']:3d}"
            f"  {s['recall_at_1']:.3f}"
            f"  {s['recall_at_3']:.3f}"
            f"  {s['recall_at_5']:.3f}"
            f"  {s['recall_at_10']:.3f}"
            f"  {s['mrr']:.3f}"
            f"  {s['map']:.3f}"
            f"  {s['ndcg_at_5']:.3f}"
            f"  {s['ndcg_at_10']:.3f}"
        )
    print(f"\nSaved results to {out_path}")
    return out


def _parse_judge_json(text: str) -> dict | None:
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def llm_judge(
    results_path: str,
    client,
    model: str | None = None,
    out_path: str | None = None,
) -> dict:
    with open(results_path) as f:
        data = json.load(f)
    out_path = out_path or results_path
    chosen_model = model or DEFAULT_MODEL

    for r in tqdm(data["results"], desc="LLM judge"):
        for mode, m in r["modes"].items():
            if not m.get("answer"):
                continue
            prompt = JUDGE_PROMPT.format(
                question=r["question"],
                ground_truth=r["ground_truth"],
                answer=m["answer"],
            )
            text = _call_with_retry(
                client,
                model=chosen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            scores = _parse_judge_json(text) or {}
            m["judge"] = {
                "correctness": scores.get("correctness"),
                "groundedness": scores.get("groundedness"),
                "completeness": scores.get("completeness"),
                "raw": text,
            }

    summary = data.setdefault("summary", {})
    summary["judge_per_mode"] = {}
    summary["judge_per_category"] = {}
    for r in data["results"]:
        cat = r.get("category", "unknown")
        for mode, m in r["modes"].items():
            j = m.get("judge")
            if not j:
                continue
            for dim in ("correctness", "groundedness", "completeness"):
                v = j.get(dim)
                if not isinstance(v, (int, float)):
                    continue
                summary["judge_per_mode"].setdefault(mode, {}).setdefault(dim, []).append(v)
                summary["judge_per_category"].setdefault(cat, {}).setdefault(dim, []).append(v)

    def _meanify(d: dict) -> dict:
        return {
            k: {dim: sum(vs) / len(vs) for dim, vs in v.items() if vs}
            for k, v in d.items()
        }
    summary["judge_per_mode"] = _meanify(summary["judge_per_mode"])
    summary["judge_per_category"] = _meanify(summary["judge_per_category"])

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print("\n=== Judge per mode ===")
    for mode, dims in summary["judge_per_mode"].items():
        parts = "  ".join(f"{d}={v:.2f}" for d, v in dims.items())
        print(f"  {mode:8s}  {parts}")
    print("\n=== Judge per category ===")
    for cat, dims in summary["judge_per_category"].items():
        parts = "  ".join(f"{d}={v:.2f}" for d, v in dims.items())
        print(f"  {cat:20s}  {parts}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval + (optional) LLM judge eval")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--out", default="eval/results.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=["text", "protein", "hybrid"])
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Run LLM judge after eval")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    retriever = Retriever()
    client = None
    if not args.skip_generate or args.judge:
        client = make_client()

    evaluate(
        args.questions,
        retriever,
        client,
        modes=tuple(args.modes),
        top_k=args.top_k,
        out_path=args.out,
        skip_generate=args.skip_generate,
        model=args.model,
    )

    if args.judge:
        llm_judge(args.out, client, model=args.model)


if __name__ == "__main__":
    main()
