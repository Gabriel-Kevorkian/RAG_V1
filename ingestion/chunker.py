"""
chunker.py  —  Day 2, Part 1: CHUNKING
======================================

Splits each cleaned document from ``document_processor.py`` into small,
overlapping, sentence-aware pieces ("chunks") of roughly 512 tokens.

Why chunk at all?
  * Retrieval should return a focused paragraph, not a 50-page class page.
  * Embedding models have input-size limits.
  * Overlap (50 tokens) means an idea that straddles a chunk boundary still
    appears whole in at least one chunk, instead of being cut in half.

Each chunk keeps its parent's metadata (source, title) and gets a unique
``chunk_id``. That id is important: on Day 3 we fuse results from the dense
and sparse indices, and we need a stable key to recognise "this is the same
chunk" across both.
"""

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.document_processor import load_documents

# ~512-token chunks with 50 tokens of overlap; drop anything tiny.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_TOKENS = 100

# One shared tokenizer so "how many tokens?" means the same thing everywhere.
# cl100k_base is a fast, standard tokenizer — we only use it to MEASURE length
# for sizing chunks, which is independent of which model embeds them later.
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text`` using our shared tokenizer."""
    return len(_encoder.encode(text))


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split each document into ~512-token overlapping chunks.

    The splitter is "recursive": it tries to break on the biggest natural
    boundary first (blank line), then a newline, then sentence endings
    (". ", "? ", "! "), then spaces — only cutting mid-word as a last resort.
    That is how we honour "never split mid-sentence" in practice.

    Chunks shorter than ``MIN_CHUNK_TOKENS`` are dropped: they are usually
    stray headers or fragments that add noise to the index.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ordered from "best" boundary to "worst" — sentence-aware.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in documents:
        pieces = splitter.split_text(doc.page_content)

        local_i = 0
        for piece in pieces:
            if count_tokens(piece) < MIN_CHUNK_TOKENS:
                continue  # skip tiny fragments

            chunk_id = f"{doc.metadata['source']}::{local_i}"
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={**doc.metadata, "chunk_id": chunk_id},
                )
            )
            local_i += 1

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Loaded {len(docs)} documents -> produced {len(chunks)} chunks\n")

    # How many chunks did each document produce? (Expect the big classes to
    # dominate and the small ones to make just a few.)
    from collections import Counter
    per_source = Counter(c.metadata["source"] for c in chunks)
    print("Chunks per document (top 5 largest):")
    for source, n in per_source.most_common(5):
        print(f"  {n:4d}  {source}")
    print(f"  ... {len(per_source)} documents total\n")

    token_counts = [count_tokens(c.page_content) for c in chunks]
    print(f"Chunk token sizes — min: {min(token_counts)}  "
          f"max: {max(token_counts)}  avg: {sum(token_counts)//len(token_counts)}")

    print("\n" + "=" * 60)
    print(f"First chunk  (id: {chunks[0].metadata['chunk_id']})")
    print("=" * 60)
    print(chunks[0].page_content[:500])
