"""
test_retrieval.py  —  Days 3 & 4: compare DENSE vs SPARSE vs HYBRID vs RERANKED
==============================================================================

Runs a few hand-picked queries through every retriever and prints the top
results of each, so you can SEE where they agree and disagree. The queries are
chosen to stress different strengths:

  * an exact method name          -> sparse (keyword) should dominate
  * a paraphrased 'how do I...'    -> dense (semantic) should dominate
  * a mixed factual lookup         -> hybrid should combine the best of both

The last block is Day 4: we take HYBRID's 20 candidates and re-order them with
a cross-encoder, which actually reads each chunk against the query. We print
BOTH the before (hybrid) and after (reranked) order so the change is visible,
plus the reranking latency in ms — the price we pay for that accuracy.

Run from the project root:  python -m scripts.test_retrieval
"""

from retrieval.dense_retriever import retrieve_dense
from retrieval.sparse_retriever import retrieve_sparse
from retrieval.hybrid_retriever import retrieve_hybrid
from retrieval.reranker import rerank

QUERIES = [
    "computeIfAbsent",                              # exact method name (keyword-heavy)
    "how do I read text from a file line by line",  # paraphrase-heavy
    "what is the default initial capacity of a HashMap",  # mixed factual lookup
]

TOP_K = 20  # candidates each retriever returns (and what the reranker sifts through)


def show(label: str, results, n: int = 3):
    print(f"  {label}")
    for i, (doc, score) in enumerate(results[:n], start=1):
        print(f"    {i}. {doc.metadata['chunk_id']:38s}  score={score:.4f}")


if __name__ == "__main__":
    for query in QUERIES:
        print("=" * 72)
        print(f"QUERY: {query!r}")
        print("=" * 72)

        show("DENSE  (semantic):", retrieve_dense(query, k=TOP_K))
        show("SPARSE (keyword): ", retrieve_sparse(query, k=TOP_K))

        hybrid = retrieve_hybrid(query, k=TOP_K)
        show("HYBRID (RRF):     ", hybrid)

        # Day 4: hand hybrid's candidates to the cross-encoder for a careful re-read.
        candidates = [doc for doc, _ in hybrid]
        reranked, latency_ms = rerank(query, candidates)
        show(f"RERANKED (cross-encoder, {len(candidates)} pairs in {latency_ms:.0f} ms):",
             reranked, n=5)
        print()
