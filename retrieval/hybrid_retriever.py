"""
hybrid_retriever.py  —  Day 3: FUSE dense + sparse with Reciprocal Rank Fusion
==============================================================================

retrieve_hybrid(query, k) runs BOTH retrievers, then merges their two ranked
lists using Reciprocal Rank Fusion (RRF). RRF ignores the raw scores (which are
on incompatible scales) and uses only each chunk's RANK in each list:

    rrf_score(chunk) = sum over lists of  1 / (RRF_K + rank_in_that_list)

A chunk that BOTH retrievers rank highly collects two contributions and rises
to the top — agreement is rewarded automatically.
"""

from langchain_core.documents import Document

from retrieval.dense_retriever import retrieve_dense
from retrieval.sparse_retriever import retrieve_sparse

RRF_K = 60  # standard constant; larger = flatter, smaller = top ranks dominate


def retrieve_hybrid(query: str, k: int = 20) -> list[tuple[Document, float]]:
    """Return the top-k chunks after RRF-fusing dense and sparse results."""
    dense = retrieve_dense(query, k=k)
    sparse = retrieve_sparse(query, k=k)

    rrf_scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}

    # Add each list's contribution: 1 / (RRF_K + rank), rank starting at 1.
    for ranked_list in (dense, sparse):
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            cid = doc.metadata["chunk_id"]
            docs_by_id[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    # Sort chunk ids by fused score, highest first, and return the top k.
    top_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:k]
    return [(docs_by_id[cid], rrf_scores[cid]) for cid in top_ids]
