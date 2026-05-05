"""Generate grounded answers from retrieved protein context using OpenAI."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()


DEFAULT_MODEL = "gpt-5.4-mini"


def make_client(api_key: str | None = None) -> OpenAI:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set")
    return OpenAI(api_key=api_key)


def _call_with_retry(client: OpenAI, max_retries: int = 6, **kwargs) -> str:
    """Call chat completions with exponential backoff on rate limit errors."""
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def _format_context(retrieved_proteins: list[dict]) -> str:
    blocks = []
    for p in retrieved_proteins:
        blocks.append(
            f"Protein: {p.get('name', '')} ({p.get('id', '')})\n"
            f"{p.get('annotation', '')}"
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, retrieved_proteins: list[dict]) -> str:
    context = _format_context(retrieved_proteins)
    return (
        "You are a protein biology expert. Answer the question below using the provided "
        "context as your primary source. You may supplement with your own knowledge when "
        "the context is insufficient, but prioritize context and do not contradict it.\n\n"
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
    return _call_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
