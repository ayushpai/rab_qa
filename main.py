"""CLI entrypoint for the RAB QA pipeline."""

import argparse

from src.generate import generate_answer, make_client
from src.retrieve import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="RAB: Retrieval Augmented Biology QA")
    parser.add_argument("--query", required=True, help="Natural-language question or amino-acid sequence")
    parser.add_argument("--sequence", default=None, help="Optional explicit amino-acid sequence for hybrid retrieval")
    parser.add_argument("--mode", choices=["auto", "text", "protein", "hybrid"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight on text score")
    parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation step")
    parser.add_argument("--model", default=None, help="Override Fireworks model id")
    args = parser.parse_args()

    print(f"\n=== Retrieving (mode={args.mode}, top_k={args.top_k}) ===")
    retriever = Retriever()
    retrieved = retriever.retrieve(
        args.query, sequence=args.sequence, mode=args.mode, top_k=args.top_k, alpha=args.alpha
    )

    for i, p in enumerate(retrieved, 1):
        score = p.get("_score", 0.0)
        print(f"{i}. [{p['id']}] score={score:.3f}  {p['name']}")

    if args.no_generate:
        return

    print("\n=== Generating answer ===")
    client = make_client()
    kwargs = {"model": args.model} if args.model else {}
    answer = generate_answer(args.query, retrieved, client, **kwargs)
    print(answer)


if __name__ == "__main__":
    main()
