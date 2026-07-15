"""
dense_retriever.py  —  Day 3: SEMANTIC search over the Chroma index
====================================================================

retrieve_dense(query, k) embeds the query with the SAME Gemini model used to
build the index, then asks ChromaDB for the k nearest chunk vectors. "Nearest"
means closest in meaning — this is the retriever that shines on paraphrased
questions where the exact words differ.
"""

import chromadb
from langchain_core.documents import Document

from ingestion.indexer import CHROMA_DIR, COLLECTION_NAME, get_embeddings

# Load the collection and the embeddings client once, then reuse them.
_collection = None
_embeddings = None


def _load():
    global _collection, _embeddings
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
        _embeddings = get_embeddings()
    return _collection, _embeddings


def retrieve_dense(query: str, k: int = 20) -> list[tuple[Document, float]]:
    """Return the top-k chunks by semantic similarity, as (Document, score) pairs.

    Chroma returns a distance (smaller = closer). We convert it to a 0..1
    'similarity' purely for readable display; the RANKING is whatever Chroma
    returns. Chunks come back already ordered nearest-first.
    """
    collection, embeddings = _load()

    query_vector = embeddings.embed_query(query)
    result = collection.query(query_embeddings=[query_vector], n_results=k)

    results: list[tuple[Document, float]] = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = 1.0 / (1.0 + distance)  # display-only: higher = closer
        results.append((Document(page_content=text, metadata=meta), similarity))
    return results
