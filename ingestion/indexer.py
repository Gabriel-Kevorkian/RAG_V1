"""
indexer.py  —  Day 2, Part 2: BUILD BOTH SEARCH INDICES
=======================================================

We build TWO independent indices over the same chunks. This is the core of
"hybrid" retrieval — each index finds relevant text a different way:

  build_dense_index(chunks)
      Converts every chunk into an embedding VECTOR with Google's Gemini
      embedding model and stores it in a persistent local ChromaDB collection
      named 'chunks'. Finds text by MEANING (semantic similarity).

  build_sparse_index(chunks)
      Builds a classic BM25 keyword index over the same chunks and pickles it
      to disk. Finds text by exact WORD overlap.

Both indices key their entries on the chunk's ``chunk_id`` so that on Day 3 we
can tell when the two indices return the very same chunk and fuse them.
"""

import os
import pickle
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from ingestion.document_processor import load_documents
from ingestion.chunker import chunk_documents, count_tokens

# Load GOOGLE_API_KEY from the .env file into the environment.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"          # dense index lives here
BM25_PATH = PROJECT_ROOT / "indices" / "bm25.pkl"  # sparse index lives here
COLLECTION_NAME = "chunks"
EMBED_MODEL = "gemini-embedding-2"                 # Gemini embedding model (free tier: 100 RPM / 30k TPM / 1k RPD)

_PLACEHOLDER = "PASTE_YOUR_KEY_HERE"


def bm25_tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens for BM25.

    Using a regex (not str.split) means punctuation doesn't stay glued to
    words: 'computeIfAbsent(K' becomes ['computeifabsent', 'k'], so an exact
    search for 'computeIfAbsent' actually matches. The SAME function is used
    when building the index and when querying it — they must agree.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def get_embeddings():
    """Create the Gemini embeddings client, with a clear error if the key is missing."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == _PLACEHOLDER:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Open the .env file and replace "
            f"'{_PLACEHOLDER}' with your real key from "
            "https://aistudio.google.com/apikey"
        )
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)


# Stay safely under the 30,000 tokens-per-minute free-tier limit.
TPM_BUDGET = 25_000
PAUSE_SECONDS = 60


def _embed_with_retry(embeddings, texts: list[str], retries: int = 4) -> list[list[float]]:
    """Embed a batch of texts, retrying with a pause if we hit a rate limit."""
    for attempt in range(retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if attempt == retries - 1:
                raise
            first_line = str(e).splitlines()[0][:90]
            print(f"    [error: {first_line}] waiting {PAUSE_SECONDS}s then retrying...", flush=True)
            time.sleep(PAUSE_SECONDS)
    return []  # unreachable, keeps type-checkers happy


def build_dense_index(chunks: list[Document]):
    """Embed all chunks with Gemini and store them in a persistent Chroma collection.

    We embed in token-budgeted batches and pause between them so we never send
    more than ~25k tokens per minute (the free tier allows 30k/min). Chroma is
    given the *precomputed* vectors, so it stores exactly what Gemini produced.

    Returns the chromadb collection.
    """
    import chromadb

    embeddings = get_embeddings()

    # A persistent, on-disk Chroma database. Rebuild the collection from scratch
    # each run so re-running never leaves stale/duplicate vectors behind.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    total = len(chunks)
    i = done = 0
    while i < total:
        # Grow a batch until adding one more chunk would exceed the token budget.
        batch: list[Document] = []
        batch_tokens = 0
        while i < total:
            t = count_tokens(chunks[i].page_content)
            if batch and batch_tokens + t > TPM_BUDGET:
                break
            batch.append(chunks[i])
            batch_tokens += t
            i += 1

        vectors = _embed_with_retry(embeddings, [c.page_content for c in batch])
        collection.add(
            ids=[c.metadata["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        done += len(batch)
        print(f"  embedded {done}/{total}  (this batch: {len(batch)} chunks, ~{batch_tokens:,} tokens)", flush=True)

        if i < total:  # more to go — wait out the rate-limit window
            time.sleep(PAUSE_SECONDS)

    return collection


def build_sparse_index(chunks: list[Document]):
    """Build a BM25 keyword index over the chunks and pickle it to disk.

    We store BOTH the BM25 object and the chunks list together, because BM25
    only returns *positions* (chunk #3 scored highest) — we need the saved
    chunks alongside it to turn a position back into real text on Day 3.
    """
    from rank_bm25 import BM25Okapi

    # Tokenize with our shared tokenizer so index + queries stay consistent.
    tokenized = [bm25_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return bm25


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"{len(chunks)} chunks to index.\n")

    # Sparse first — it is instant and needs no API, so it never wastes calls.
    print("Building SPARSE (BM25) index...")
    build_sparse_index(chunks)
    print(f"  saved -> {BM25_PATH}\n")

    # Dense second — this is the step that calls the Gemini API (paced, ~12-15 min).
    print("Building DENSE (Chroma + Gemini embeddings) index... [calls the API, paced]")
    collection = build_dense_index(chunks)
    stored = collection.count()
    print(f"  stored {stored} vectors in Chroma collection '{COLLECTION_NAME}' -> {CHROMA_DIR}")

    print("\nBoth indices built. Ready for Day 3 (retrieval).")
