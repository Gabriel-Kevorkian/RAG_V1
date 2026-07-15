"""
sparse_retriever.py  —  Day 3: KEYWORD search over the BM25 index
=================================================================

retrieve_sparse(query, k) loads the pickled BM25 index and returns the k chunks
whose words best overlap the query (BM25 scoring). This is the retriever that
shines on exact terms: method names, IDs, error codes.
"""

import pickle

from langchain_core.documents import Document

from ingestion.indexer import BM25_PATH, bm25_tokenize

# Load the pickled {bm25, chunks} once, then reuse.
_data = None


def _load():
    global _data
    if _data is None:
        with open(BM25_PATH, "rb") as f:
            _data = pickle.load(f)
    return _data


def retrieve_sparse(query: str, k: int = 20) -> list[tuple[Document, float]]:
    """Return the top-k chunks by BM25 keyword score, as (Document, score) pairs."""
    data = _load()
    bm25, chunks = data["bm25"], data["chunks"]

    # Score every chunk against the query, using the SAME tokenizer as indexing.
    scores = bm25.get_scores(bm25_tokenize(query))

    # Indices of the k highest-scoring chunks, best first.
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]
