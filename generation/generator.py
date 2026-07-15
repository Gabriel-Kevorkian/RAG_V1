"""
generator.py  —  Day 5: turn retrieved chunks into a grounded ANSWER
====================================================================

This is the "G" in RAG. Retrieval (Days 3-4) found the best 5 chunks; now an
LLM has to write an answer from them.

The whole difficulty here is that Gemini ALREADY knows Java from pretraining.
If we just say "here are some chunks, answer the question", it will cheerfully
answer from memory and silently mix in details that are not in our corpus --
a hallucination we would have no way to detect.

So the prompt is deliberately restrictive. Three rules do the work:

  1. Answer ONLY from the context.       -> stops it from using pretrained memory
  2. Cite the source filename.           -> makes every claim traceable to a chunk
  3. Say "insufficient context" if the   -> gives it a legal way to refuse, so it
     answer isn't there.                    doesn't have to invent one

Rule 3 is the one beginners skip. Without an explicit escape hatch, an LLM asked
an unanswerable question will ALWAYS produce something -- and that something will
look confident. Day 6-7's "unanswerable" test questions exist to check exactly
this behaviour, and they only mean anything because of rule 3.
"""

import os
import time

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

GEN_MODEL = "gemini-3.1-flash-lite"  # free tier; verified working
INSUFFICIENT = "insufficient context"

SYSTEM_PROMPT = f"""You are a precise Java API reference assistant.

Answer the user's question using ONLY the context passages provided below.

Rules:
1. Use ONLY facts stated in the context. Never use prior knowledge about Java,
   even if you are confident it is correct.
2. Cite the source file for every claim, in square brackets, e.g. [java.util.HashMap.html].
3. If the context does not contain the answer, reply exactly: "{INSUFFICIENT}"
   and nothing else. Do not guess, and do not partially answer.
4. Be concise and concrete. Prefer exact signatures, names, and values from the
   context over paraphrase.
"""

_client = None


def _get_client():
    """Create the Gemini client once and reuse it."""
    global _client
    if _client is None:
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing from .env")
        _client = genai.Client(api_key=api_key)
    return _client


def build_context(chunks: list[Document]) -> str:
    """Format the retrieved chunks into a labelled context block.

    Each passage is tagged with its source filename. That label is not decoration:
    it is the ONLY way the model can obey rule 2 and cite anything. If we stripped
    the filenames, citations would become impossible and we would lose our ability
    to check the answer against the corpus.
    """
    blocks = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[Passage {i} | source: {source}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def generate(query: str, chunks: list[Document]) -> tuple[str, float]:
    """Ask the LLM to answer `query` using only `chunks`.

    Returns (answer_text, latency_ms). Latency is tracked because Day 8's
    ablation table reports end-to-end P50/P95 timings per configuration.
    """
    if not chunks:
        return INSUFFICIENT, 0.0

    client = _get_client()

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{build_context(chunks)}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ==="
    )

    start = time.perf_counter()
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        # temperature=0 -> as close to deterministic as we can get. For a factual
        # reference lookup we want the SAME answer every run; creativity here is
        # just another word for hallucination. It also makes evaluation stable:
        # if Day 8's score moves, it moved because the pipeline changed, not
        # because the model rolled different dice.
        config={"temperature": 0.0},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    answer = (response.text or "").strip()
    return answer, latency_ms
