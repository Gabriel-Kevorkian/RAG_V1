"""
ask.py  —  Day 5: ask the RAG system a question from the command line
=====================================================================

Usage (from the project root):

    python -m scripts.ask "what is the default initial capacity of a HashMap"
    python -m scripts.ask --mode dense_only "computeIfAbsent"
    python -m scripts.ask                      # interactive: keeps asking

This is the first time the whole system runs end to end. Things worth watching:

  * SOURCES  -- which corpus files the answer was built from. If the answer cites
                a file that couldn't possibly contain it, retrieval failed.
  * TIMINGS  -- where the seconds actually go (spoiler: reranking + generation).
  * Try a question the corpus CANNOT answer, e.g. "who invented Java?".
    A working system says "insufficient context". A broken one confidently
    answers "James Gosling" -- correct, but from the model's memory, not from
    our corpus. That is a hallucination, and it is exactly what we are guarding
    against.
"""

import sys

from generation.pipeline import answer, MODES
from generation.generator import INSUFFICIENT


def show(result: dict) -> None:
    print()
    print("=" * 72)
    print(f"Q: {result['question']}   [mode: {result['mode']}]")
    print("=" * 72)
    print(result["answer"])
    print()

    if result["answer"].strip().lower().startswith(INSUFFICIENT):
        print("  (the system refused rather than guessed — that is the correct behaviour)")

    print("  SOURCES USED:")
    for source in result["sources"]:
        print(f"    - {source}")

    t = result["timings"]
    print(
        f"  TIMINGS: retrieval {t['retrieval_ms']:.0f} ms | "
        f"rerank {t['rerank_ms']:.0f} ms | "
        f"generation {t['generation_ms']:.0f} ms | "
        f"TOTAL {t['total_ms']:.0f} ms"
    )
    print()


def main() -> None:
    args = sys.argv[1:]

    mode = "hybrid_rerank"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        if mode not in MODES:
            sys.exit(f"unknown mode {mode!r}; expected one of {MODES}")
        del args[i : i + 2]

    if args:  # one-shot: question given on the command line
        show(answer(" ".join(args), mode=mode))
        return

    # interactive loop
    print(f"RAG ready (mode: {mode}). Ask a Java API question, or Ctrl-C to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if question:
            show(answer(question, mode=mode))


if __name__ == "__main__":
    main()
