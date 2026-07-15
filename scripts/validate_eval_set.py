"""
validate_eval_set.py  —  Days 6-7: check your eval set before you trust it
=========================================================================

    python -m scripts.validate_eval_set

The eval set is the RULER you will measure the whole system with on Day 8. A bug
in the ruler is worse than a bug in the system, because it produces confident,
believable, wrong numbers. So we check it mechanically:

  * every required field is present, and query_type is one of the four
  * ids are unique
  * the counts roughly match the guide's targets
  * ANSWERABLE questions: the source_file exists in the corpus, AND the
    source_passage really appears (verbatim) in that file's indexed chunks
  * UNANSWERABLE questions: answer is "insufficient context", no source

That fourth check is the important one. It proves the evidence for each answer
is actually IN the corpus and REACHABLE by the retriever. If a passage doesn't
appear in any chunk, then no retriever could ever find it -- the question is
unfair, and it would show up on Day 8 as a "system failure" that is really an
eval-set bug.
"""

import json
import re
import sys
from pathlib import Path

from retrieval.sparse_retriever import _load
from generation.generator import INSUFFICIENT
from evaluation.gold import golds

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"

QUERY_TYPES = ["simple", "keyword_heavy", "paraphrase_heavy", "unanswerable"]
TARGETS = {              # from the build guide
    "simple": (20, 25),
    "keyword_heavy": (10, 10),
    "paraphrase_heavy": (10, 10),
    "unanswerable": (10, 10),
}
REQUIRED = ["id", "question", "ground_truth_answer", "source_passage", "source_file", "query_type"]


def normalize(text: str) -> str:
    """Collapse whitespace so a copied passage still matches despite line breaks."""
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    examples = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if not examples:
        sys.exit(f"{EVAL_PATH} is empty — write your examples first "
                 f"(see eval/eval_set.example.json for the format).")

    chunks = _load()["chunks"]
    corpus_files = {c.metadata["source"] for c in chunks}
    # All chunk text per file, normalized, so we can check a passage really occurs.
    text_by_file: dict[str, str] = {}
    for c in chunks:
        src = c.metadata["source"]
        text_by_file[src] = text_by_file.get(src, "") + " " + normalize(c.page_content)

    errors: list[str] = []
    counts = {t: 0 for t in QUERY_TYPES}
    seen_ids: set[str] = set()

    for i, ex in enumerate(examples):
        tag = f"[{i}] {ex.get('id', '?')}"

        missing = [f for f in REQUIRED if f not in ex]
        if missing:
            errors.append(f"{tag}: missing field(s): {', '.join(missing)}")
            continue

        if ex["id"] in seen_ids:
            errors.append(f"{tag}: duplicate id")
        seen_ids.add(ex["id"])

        qtype = ex["query_type"]
        if qtype not in QUERY_TYPES:
            errors.append(f"{tag}: bad query_type {qtype!r}; expected one of {QUERY_TYPES}")
            continue
        counts[qtype] += 1

        if not ex["question"].strip():
            errors.append(f"{tag}: empty question")

        if qtype == "unanswerable":
            if normalize(ex["ground_truth_answer"]) != normalize(INSUFFICIENT):
                errors.append(f"{tag}: unanswerable answer must be exactly {INSUFFICIENT!r}")
            if ex["source_file"] or ex["source_passage"]:
                errors.append(f"{tag}: unanswerable must have empty source_file and source_passage")
            continue

        # --- answerable questions ---
        # A question may have MORE THAN ONE right answer. "How do I join strings
        # with a separator?" is answered correctly by both Collectors.joining and
        # String.join, and the first version of this eval set wrote down only one
        # of them -- which meant a correct answer from the other source scored 0.00
        # and got filed as a retriever failure. Any alternate golds live in
        # `alt_golds` and are held to EXACTLY the same standard as the primary:
        # the file must be in the corpus and the passage must appear verbatim.
        # An unverified alternate gold is just a new way to lie to yourself.
        for gi, g in enumerate(golds(ex)):
            gtag = tag if gi == 0 else f"{tag} alt_golds[{gi - 1}]"

            if not g["ground_truth_answer"].strip():
                errors.append(f"{gtag}: empty ground_truth_answer")

            src = g["source_file"]
            if src not in corpus_files:
                errors.append(f"{gtag}: source_file {src!r} is not in the corpus")
                continue

            passage = normalize(g["source_passage"])
            if not passage:
                errors.append(f"{gtag}: empty source_passage")
            elif passage not in text_by_file[src]:
                errors.append(
                    f"{gtag}: source_passage not found verbatim in {src} — "
                    f"copy it exactly from `python -m scripts.browse_corpus`"
                )

    # ---- report ----
    print(f"{len(examples)} examples in {EVAL_PATH.name}\n")
    print("counts by query_type:")
    for t in QUERY_TYPES:
        lo, hi = TARGETS[t]
        ok = "ok " if lo <= counts[t] <= hi else "OFF"
        print(f"  {ok} {t:18s} {counts[t]:3d}   (target {lo}-{hi})")
    total_lo = sum(lo for lo, _ in TARGETS.values())
    total_hi = sum(hi for _, hi in TARGETS.values())
    print(f"      {'TOTAL':18s} {len(examples):3d}   (target {total_lo}-{total_hi})\n")

    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")

    short = [t for t in QUERY_TYPES if not TARGETS[t][0] <= counts[t] <= TARGETS[t][1]]
    if short:
        print(f"counts are off target for: {', '.join(short)} — keep writing.")

    if errors or short:
        sys.exit(1)

    print("No problems found. The eval set is ready for Day 8.")


if __name__ == "__main__":
    main()
