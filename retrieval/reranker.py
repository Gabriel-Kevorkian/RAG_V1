"""
reranker.py  —  Day 4: CROSS-ENCODER re-ranking
===============================================

The retrievers (dense / sparse / hybrid) are BI-ENCODERS: they compare a query
vector to chunk vectors that were computed long before the query existed. Fast,
but the chunk never actually "sees" the question.

A CROSS-ENCODER is different. It takes the pair

    [query  [SEP]  chunk_text]

as ONE input and runs it through a transformer, so every word of the question
can attend to every word of the chunk. The output is a single relevance score:
"how well does this chunk answer THIS query?" Much more accurate — but nothing
can be precomputed, so it costs one forward pass per (query, chunk) pair.

That is why we run it as stage 2 of a two-stage pipeline:

    759 chunks --[hybrid retrieval: fast]--> 20 candidates
                --[cross-encoder: precise]--> top 5

Reranking all 759 chunks would take ~10s per query; reranking 20 takes ~0.2s.

The model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is a small BERT trained on
MS MARCO — real Bing queries paired with human-judged passages. It runs locally
on the CPU: no API key, no rate limit, no cost.

NOTE: its score is a raw logit (roughly -11..+11), not a probability. Only the
ORDERING is meaningful; don't read the absolute number as a percentage.
"""

import time

from langchain_core.documents import Document

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N = 5  # how many chunks survive reranking and get sent to the LLM on Day 5

# The model is a few hundred MB to load, so build it once and reuse it.
_model = None


def _load():
    """Load the cross-encoder once (downloads ~80MB on first run, then cached)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(
    query: str,
    chunks: list[Document],
    top_n: int = TOP_N,
) -> tuple[list[tuple[Document, float]], float]:
    """Re-order `chunks` by true relevance to `query`; keep the best `top_n`.

    Returns (results, latency_ms) where results is a list of (Document, score)
    sorted best-first. We return the latency because Day 8's ablation table has
    to report what reranking costs — accuracy is never free, and the whole point
    of the ablation is to show whether the extra milliseconds buy better answers.
    """
    if not chunks:
        return [], 0.0

    model = _load()

    # Build one (query, chunk_text) PAIR per candidate. This pairing is the
    # thing that makes it a cross-encoder — contrast with embeddings, where the
    # query and the chunk are encoded separately and never meet.
    pairs = [(query, doc.page_content) for doc in chunks]

    start = time.perf_counter()
    scores = model.predict(pairs)  # one forward pass per pair
    latency_ms = (time.perf_counter() - start) * 1000

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    results = [(doc, float(score)) for doc, score in ranked[:top_n]]
    return results, latency_ms
