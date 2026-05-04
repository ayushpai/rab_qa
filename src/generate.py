"""Generate grounded answers from retrieved protein context using Fireworks AI."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"


def make_client(api_key: str | None = None) -> OpenAI:
    api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY env var not set")
    return OpenAI(base_url=FIREWORKS_BASE_URL, api_key=api_key)


def _format_context(retrieved_proteins: list[dict]) -> str:
    blocks = []
    for p in retrieved_proteins:
        blocks.append(
            f"Protein: {p.get('name', '')} ({p.get('id', '')})\n"
            f"Organism: {p.get('organism', '')}\n"
            f"Function: {p.get('function', '')}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, retrieved_proteins: list[dict]) -> str:
    context = _format_context(retrieved_proteins)
    return (
        "You are a protein biology expert. Answer the question below using ONLY the "
        "provided context. If the context does not contain enough information, say so "
        "explicitly. Do not hallucinate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_answer(
    query: str,
    retrieved_proteins: list[dict],
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    prompt = build_prompt(query, retrieved_proteins)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
