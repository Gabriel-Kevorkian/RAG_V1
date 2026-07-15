"""
gold.py  —  one question can have more than one right answer
============================================================

The eval set originally allowed exactly one gold source per question. That is a
reasonable default and it is wrong for a real corpus. "How do I glue a bunch of
strings together with a separator?" is answered correctly by `Collectors.joining`
AND by `String.join`. "How can I read a file one line at a time?" is answered
correctly by `BufferedReader.readLine` AND by `Files.lines`. The eval set wrote
down one of each pair, so when the system produced the other -- correctly,
faithfully, with a citation -- it was scored 0.00 on context_recall and filed as
a RETRIEVER FAILURE. Two of the four "failures" in the Day 10 report were this.

A single-gold eval set does not just mismeasure those questions. It actively
points the work at the wrong subsystem: it would have had me tune a retriever
that was doing its job perfectly.

So a question carries a primary gold (the top-level fields, unchanged, so nothing
else has to know about this) plus an optional `alt_golds` list. Everything that
scores retrieval takes the BEST accepted gold -- max over references, the standard
multi-reference approach -- because the question is "did the system find A right
answer?", not "did it find MY right answer?".
"""

GOLD_FIELDS = ("ground_truth_answer", "source_passage", "source_file")


def golds(example: dict) -> list[dict]:
    """Every accepted gold for a question, primary first.

    Unanswerable questions have no gold and return an empty list.
    """
    if example.get("query_type") == "unanswerable":
        return []
    primary = {f: example[f] for f in GOLD_FIELDS}
    return [primary] + [
        {f: alt[f] for f in GOLD_FIELDS} for alt in example.get("alt_golds", [])
    ]


def references(example: dict) -> list[str]:
    """The RAGAS `reference` string for each accepted gold.

    Mirrors what stage 1 used to bake into raw_runs.json: the ground-truth answer
    followed by the source passage it came from.
    """
    return [f"{g['ground_truth_answer']} {g['source_passage']}".strip()
            for g in golds(example)]
