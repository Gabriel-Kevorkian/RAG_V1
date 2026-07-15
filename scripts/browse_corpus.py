"""
browse_corpus.py  —  Days 6-7: read the corpus while you write the eval set
===========================================================================

Writing 50+ ground-truth answers means reading a lot of JavaDoc. Rather than
opening raw HTML files, this searches the CHUNKS the RAG system actually indexed
and prints them in full, so you can copy a `source_passage` verbatim.

Reading the chunks (not the HTML) matters: it shows you the corpus exactly as
the system sees it. If a fact got split across a chunk boundary, or was dropped
during cleaning, you will find out HERE -- while writing the question -- instead
of being confused on Day 8 when the system fails to answer it.

Usage (from the project root):

    python -m scripts.browse_corpus "initial capacity"        # keyword search
    python -m scripts.browse_corpus "readLine" --n 5          # more results
    python -m scripts.browse_corpus --file HashMap            # list one class's chunks
    python -m scripts.browse_corpus --file HashMap --chunk 5  # print one chunk in full

TIP for writing good questions: pick a passage FIRST, then write a question that
only that passage can answer. It is much easier than inventing a question and
then hunting for evidence.
"""

import sys

from retrieval.sparse_retriever import retrieve_sparse, _load


def print_chunk(doc, score: float | None = None, full: bool = True) -> None:
    cid = doc.metadata["chunk_id"]
    header = f"--- {cid}"
    if score is not None:
        header += f"   (bm25 score {score:.1f})"
    print(header + " " + "-" * max(0, 68 - len(header)))
    text = doc.page_content
    print(text if full else text[:300] + ("..." if len(text) > 300 else ""))
    print()


def main() -> None:
    args = sys.argv[1:]

    if "--file" in args:
        i = args.index("--file")
        needle = args[i + 1].lower()
        chunks = [
            c for c in _load()["chunks"]
            if needle in c.metadata["source"].lower()
        ]
        if not chunks:
            sys.exit(f"no corpus file matching {needle!r}")

        if "--chunk" in args:
            n = args[args.index("--chunk") + 1]
            match = [c for c in chunks if c.metadata["chunk_id"].endswith(f"::{n}")]
            if not match:
                sys.exit(f"no chunk ::{n} in that file")
            print_chunk(match[0])
            return

        print(f"{len(chunks)} chunks in {chunks[0].metadata['source']}:\n")
        for c in chunks:
            print_chunk(c, full=False)
        return

    if not args:
        sys.exit(__doc__)

    n = 3
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        del args[i : i + 2]

    query = " ".join(args)
    print(f"top {n} chunks for {query!r}:\n")
    for doc, score in retrieve_sparse(query, k=n):
        print_chunk(doc, score)


if __name__ == "__main__":
    main()
