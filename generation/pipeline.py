"""
pipeline.py  —  Day 5: the full RAG pipeline, end to end
========================================================

    question -> retrieve -> (rerank) -> generate -> grounded answer

The `mode` argument selects which retrieval strategy to use:

    dense_only     -> semantic search only
    sparse_only    -> BM25 keyword search only
    hybrid         -> RRF fusion of both
    hybrid_rerank  -> RRF fusion, then cross-encoder rerank   (the full system)

Those four names are not arbitrary: they are exactly the four rows of the Day 8
ablation table. An ablation means "remove one piece at a time and measure what
breaks" -- it is how you prove each component actually earns its place, instead
of just assuming it does. Building the switch NOW means Day 8 is a for-loop
rather than a rewrite.

Every stage is timed, because "better answers" is only half the story: Day 8
also has to report what each configuration COSTS (P50/P95 latency).
"""

import time

from langchain_core.documents import Document

from retrieval.dense_retriever import retrieve_dense
from retrieval.sparse_retriever import retrieve_sparse
from retrieval.hybrid_retriever import retrieve_hybrid
from retrieval.reranker import rerank, TOP_N
from generation.generator import generate

RETRIEVE_K = 20  # wide net: recall matters here, precision is the reranker's job
MODES = ["dense_only", "sparse_only", "hybrid", "hybrid_rerank"]


def retrieve(question: str, mode: str = "hybrid_rerank") -> tuple[list[Document], dict]:
    """Run the retrieval half of the pipeline. Returns (chunks, timings_ms).

    Note the non-rerank modes are truncated to TOP_N as well, so that every mode
    hands the SAME NUMBER of chunks to the LLM. Otherwise the ablation would be
    unfair: a mode feeding 20 chunks would look better simply because it fed more
    context, and we would learn nothing about the retrieval strategy itself.
    Change one variable at a time -- that is the whole discipline of an ablation.
    """
    timings: dict[str, float] = {}

    start = time.perf_counter()
    if mode == "dense_only":
        scored = retrieve_dense(question, k=RETRIEVE_K)
    elif mode == "sparse_only":
        scored = retrieve_sparse(question, k=RETRIEVE_K)
    elif mode in ("hybrid", "hybrid_rerank"):
        scored = retrieve_hybrid(question, k=RETRIEVE_K)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    timings["retrieval_ms"] = (time.perf_counter() - start) * 1000

    candidates = [doc for doc, _score in scored]

    if mode == "hybrid_rerank":
        reranked, rerank_ms = rerank(question, candidates, top_n=TOP_N)
        chunks = [doc for doc, _score in reranked]
        timings["rerank_ms"] = rerank_ms
    else:
        chunks = candidates[:TOP_N]
        timings["rerank_ms"] = 0.0

    return chunks, timings


def answer(question: str, mode: str = "hybrid_rerank") -> dict:
    """Run the complete pipeline and return the answer plus everything we used.

    We return the CHUNKS too, not just the text. Day 8's RAGAS metrics need them
    (context_precision / context_recall are computed over the retrieved chunks),
    and on Day 10 they are how you debug a bad answer: was the right passage
    never retrieved, or was it retrieved and then ignored? Those are completely
    different bugs with completely different fixes.
    """
    chunks, timings = retrieve(question, mode=mode)
    text, gen_ms = generate(question, chunks)
    timings["generation_ms"] = gen_ms
    timings["total_ms"] = sum(timings.values())

    return {
        "question": question,
        "mode": mode,
        "answer": text,
        "chunks": chunks,
        "sources": sorted({c.metadata.get("source", "unknown") for c in chunks}),
        "timings": timings,
    }
