"""
failure_analysis.py  —  Day 10: stop reading means, start reading failures
==========================================================================

The ablation table gives four numbers per config. Numbers tell you THAT the
system is imperfect; they never tell you WHY, and "faithfulness 0.93" is not
something you can go and fix. This script opens up the per-question scores and
sorts the failures into buckets that each imply a DIFFERENT repair.

The central idea, and the reason both context_recall and faithfulness exist:

    evidence not retrieved  -> the LLM was set up to fail. FIX THE RETRIEVER.
    evidence retrieved,
    answer drifted anyway   -> FIX THE PROMPT / THE MODEL.

From the outside these look identical: a wrong answer. They are opposite bugs,
and if you only ever look at the mean you will spend a week tuning the retriever
for a problem that lives in the prompt.

WHAT CHANGED ON THIS PASS (and why the first version of this file lied to me)
----------------------------------------------------------------------------
v1 bucketed failures using RAGAS's context_recall, and measured retrieval with a
FILE-level hit: did the gold JavaDoc file appear among the retrieved chunks?
Both choices were wrong, and they were wrong in a way that flattered the system.

  * File-level hit is far too coarse. `java.util.TreeSet.html` is a 200KB page.
    Retrieving *some* chunk of it is not the same as retrieving the ONE sentence
    that answers the question. v1 reported source_hit = 1.000 for the full system
    and called it a triumph. Measured at the passage level -- using the
    `source_passage` field that has been sitting in eval_set.json unused this
    whole time -- the real number is 0.911. Four questions retrieved the right
    file and missed the right sentence. That is where the failures actually live.

  * context_recall is an LLM judge score, and on this corpus it is noisy enough
    to invert the diagnosis. It put p07 (a correct refusal after a retrieval miss)
    in the LLM bucket and p06/p10 in the retriever bucket when neither belonged
    there. Buckets built on it sent the work to the wrong subsystem.

So the buckets below are now built on passage_hit -- a judge-free, zero-API,
string-level fact -- and on whether the model refused. RAGAS scores are still
reported, but they no longer decide the diagnosis. They are evidence, not verdict.

We also finally score the 10 `unanswerable` questions. RAGAS silently drops them
(no reference to score against), so 10 of our 55 questions -- the ones that test
whether the system HALLUCINATES when the corpus has no answer -- were invisible
in every previous report.

Everything here is computed from cached files. ZERO API calls.

    python -m scripts.failure_analysis
    python -m scripts.failure_analysis --mode hybrid --worst 10
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from evaluation.gold import golds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_PATH = PROJECT_ROOT / "eval" / "eval_set.json"
REPORT_PATH = RESULTS_DIR / "failure_analysis.md"

MODES = ["dense_only", "sparse_only", "hybrid", "hybrid_rerank"]
FULL_SYSTEM = "hybrid_rerank"

# RAGAS names the column after the metric CLASS, not the metric.
PRECISION_COL = "llm_context_precision_with_reference"

# Below this, we call it a failure. 0.5 is a deliberate choice: these metrics are
# roughly "what fraction of claims / chunks were good", so <0.5 means the majority
# of the thing being measured was wrong. Not a tuned number, just an honest line.
FAIL = 0.5

# The generator is prompted to emit exactly this when the context does not support
# an answer. Detecting it is how we tell a refusal apart from a wrong answer --
# RAGAS cannot, and scores a correct refusal as a total failure (faithfulness 0,
# answer_relevancy 0), which is how p07 ended up misfiled.
REFUSAL = "insufficient context"

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def norm(s: str) -> str:
    """Collapse to comparable text. The JavaDoc HTML is full of &nbsp; and mojibake;
    exact substring matching without this is a coin flip."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def is_refusal(response: str) -> bool:
    return str(response).strip().lower().startswith(REFUSAL)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two frames: the 45 answerable questions RAGAS scored, and all 55 raw runs
    (including the 10 unanswerable ones RAGAS refuses to touch)."""
    raw = json.loads((RESULTS_DIR / "raw_runs.json").read_text(encoding="utf-8"))
    gold = {e["id"]: e for e in json.loads(EVAL_PATH.read_text(encoding="utf-8"))}

    # --- every run, judge or no judge ---------------------------------------
    runs = []
    for mode in MODES:
        for qid, g in gold.items():
            key = f"{mode}::{qid}"
            if key not in raw:
                continue
            run = raw[key]
            answerable = g["query_type"] != "unanswerable"
            runs.append(
                dict(
                    mode=mode,
                    id=qid,
                    query_type=g["query_type"],
                    answerable=answerable,
                    refused=is_refusal(run["answer"]),
                    # A question can have several accepted golds (evaluation/gold.py).
                    # The question is "did the system find A right answer?", not "did it
                    # find the first one I wrote down" -- so ANY accepted gold counts.
                    # file-level: did ANY chunk of an acceptable document show up?
                    file_hit=(
                        any(gg["source_file"] in run["sources"] for gg in golds(g))
                        if answerable else None
                    ),
                    # passage-level: did ONE sentence that answers it show up?
                    passage_hit=(
                        any(norm(gg["source_passage"]) in norm(" ".join(run["contexts"]))
                            for gg in golds(g))
                        if answerable else None
                    ),
                    answer=run["answer"],
                    sources=run["sources"],
                    gold_source=" | ".join(gg["source_file"] for gg in golds(g)),
                    gold_passage=" | ".join(gg["source_passage"] for gg in golds(g)),
                )
            )
    runs_df = pd.DataFrame(runs)

    # --- the judged subset ---------------------------------------------------
    frames = []
    for mode in MODES:
        path = RESULTS_DIR / f"ragas_{mode}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).rename(columns={PRECISION_COL: "context_precision"})
        df["mode"] = mode
        frames.append(df)
    scored = pd.concat(frames, ignore_index=True)
    # `query_type` lives in both frames; drop it here so the merge doesn't
    # suffix it into query_type_x / query_type_y.
    scored = scored.merge(
        runs_df.drop(columns=["answer", "query_type"]), on=["mode", "id"], how="left"
    )
    return scored, runs_df


def diagnose(r: pd.Series) -> str:
    """The bucket. Built on facts (was the gold passage there? did it refuse?), not
    on judge scores -- because the judge scores are what got this wrong last time.

    Note what this function deliberately REFUSES to decide. When the model answers
    without the gold passage, there are two possible worlds: the eval set wrote down
    too narrow a gold label and the answer is fine (p01, p10), or the model is simply
    wrong (p06). Telling those apart means comparing the answer to the ground truth,
    which is a judge call -- and a judge call is the thing we are trying not to trust
    blindly. So the bucket says ADJUDICATE and a human reads the chunks. A heuristic
    that admits what it cannot see beats one that guesses and sounds confident.
    """
    if not bool(r["passage_hit"]):
        if r["refused"]:
            # The evidence was missing and the model said so. This is the system
            # working. RAGAS scores it 0.0 / 0.0 and calls it the worst answer in
            # the set. The bug is real, but it is a RETRIEVAL bug.
            return "retrieval miss (model correctly refused)"
        return "answered without the gold passage -- ADJUDICATE"
    # Passage WAS retrieved.
    if r["faithfulness"] < FAIL:
        # Terse answers break RAGAS's claim decomposition: there is no claim to
        # verify, so faithfulness collapses to 0 on answers that are perfectly
        # grounded. Flag, do not convict.
        if len(str(r["response"]).split()) <= 12:
            return "grounding failure? (terse answer -- suspect judge artifact)"
        return "grounding failure (evidence was there, answer drifted)"
    return "ok"


def section(out: list[str], title: str) -> None:
    out.append(f"\n## {title}\n")


def main(mode: str, worst: int) -> None:
    scored, runs = load()
    out: list[str] = [
        "# Failure analysis",
        "",
        "_Buckets are computed from passage-level retrieval facts and refusal "
        "detection, not from RAGAS judge scores. See the header of "
        "`scripts/failure_analysis.py` for why the previous version's buckets were "
        "inverted._",
    ]

    sub = scored[scored["mode"] == mode].copy()
    sub["diagnosis"] = sub.apply(diagnose, axis=1)

    # ---------------------------------------------------------------- 0
    # The 10 questions no report has ever mentioned.
    section(out, "0. Abstention: does it refuse when the corpus cannot answer?")
    un = runs[~runs["answerable"]]
    tab = un.groupby("mode")["refused"].agg(["sum", "count"]).reindex(MODES)
    lines = [
        f"- `{m}` — refused **{int(r['sum'])} / {int(r['count'])}**"
        for m, r in tab.iterrows()
    ]
    out.append("\n".join(lines))
    out.append(
        "\nThe eval set has **55** questions. RAGAS scores **45** of them: it needs a "
        "`reference` to compare against, and the 10 `unanswerable` questions have none, "
        "so it drops them silently. Every previous version of *this report* inherited that "
        "blind spot and analysed 45 questions while believing it had analysed the set.\n\n"
        "To be precise about who missed what: `ablation_runner.py` never missed it. It "
        "computes `refusal_rate_unanswerable` and has been printing **1.0** in "
        "`ablation_table.csv` the whole time. It was the *failure analysis* — the file whose "
        "entire job is to ask what went wrong — that quietly scoped itself to the judged "
        "subset and never looked at the other ten.\n\n"
        "Those 10 ask what the system does when the answer is simply not in the corpus — the "
        "difference between a search box and a liar. **All four configurations refused all "
        "ten.** Zero hallucinations, including from `sparse_only`, which is the weakest "
        "retriever here and had the most opportunity to bluff."
    )

    # ---------------------------------------------------------------- 1
    section(out, "1. Retrieval, measured honestly (file-level vs passage-level)")
    # These columns are None for the unanswerable rows, which makes the dtype object,
    # which makes `~col` do INTEGER negation (-1, -2) instead of boolean not -- and
    # every row comes back truthy. Cast once, here, where the Nones are gone.
    ans = runs[runs["answerable"]].copy()
    ans[["file_hit", "passage_hit"]] = ans[["file_hit", "passage_hit"]].astype(bool)
    grid = (
        ans.groupby("mode")[["file_hit", "passage_hit"]]
        .mean()
        .round(3)
        .reindex(MODES)
    )
    out.append("```\n" + grid.to_string() + "\n```")
    out.append(
        "\n`file_hit` = an acceptable JavaDoc **file** appeared among the retrieved chunks.\n"
        "`passage_hit` = an acceptable gold **sentence** (`source_passage` / `alt_golds`, "
        "which have been in `eval_set.json` all along) appeared in the retrieved text.\n\n"
        "The gap between these two columns is the whole failure analysis. The full system "
        f"scores a perfect **{grid.loc[mode, 'file_hit']:.3f} on file_hit** — which is what "
        f"the old report proudly printed — and **{grid.loc[mode, 'passage_hit']:.3f} on "
        "passage_hit**. `java.util.TreeSet.html` is a very large page; pulling *a* chunk of "
        "it is not the same as pulling the *one sentence* that answers the question. "
        "File-level hit rate is a metric that cannot fail, and a metric that cannot fail is "
        "not measuring anything."
    )
    fc = ans[(ans["mode"] == mode) & ans["file_hit"] & ~ans["passage_hit"]]
    out.append(
        f"\nRight file, wrong sentence, in `{mode}`: "
        f"**{', '.join('`%s`' % i for i in fc['id'])}** — {len(fc)} of {len(ans[ans['mode']==mode])}. "
        "Every one of these was invisible to the old file-level metric."
    )

    # ---------------------------------------------------------------- 2
    section(out, "2. Diagnosis: whose fault is each failure?")
    bad = sub[sub["diagnosis"] != "ok"]
    counts = bad["diagnosis"].value_counts()
    out.append(
        "\n".join(f"- **{n}** — {d}" for d, n in counts.items())
        + f"\n- **{len(sub) - len(bad)}** — ok\n"
    )
    out.append(
        "These take opposite repairs. Retrieval misses want better chunking and a bigger "
        "candidate pool. Grounding failures want a stricter prompt or a stronger model. "
        "Bad gold labels want an eval-set fix and no code change at all. Tuning the "
        "retriever to fix a prompt bug is a week you do not get back — and the previous "
        "version of this file would have sent me to do exactly that.\n\n"
        "`ADJUDICATE` is not a hedge, it is the honest output of a judge-free test. When the "
        "model answers without the gold passage, the string comparison cannot tell me whether "
        "the gold label was too narrow or the model was wrong — that needs a human to read the "
        "chunks. §3 is that reading. It ran on three such questions: two (`p01`, `p10`) were "
        "bad gold labels and have been fixed in the eval set, which is why they no longer "
        "appear above. The one that remains, `p06`, is a genuine model error.\n"
    )

    for _, r in bad.sort_values("diagnosis").iterrows():
        out.append(
            f"\n**`{r['id']}`** ({r['query_type']}) — _{r['diagnosis']}_\n"
            f"- Q: {r['user_input']}\n"
            f"- gold: `{r['gold_source']}` → \"{r['gold_passage']}\"\n"
            f"- retrieved: {r['sources']} | passage_hit **{r['passage_hit']}**\n"
            f"- A: {str(r['response'])[:200]}\n"
            f"- scores: faith {r['faithfulness']:.2f}, relevancy {r['answer_relevancy']:.2f}, "
            f"precision {r['context_precision']:.2f}, recall {r['context_recall']:.2f}"
        )

    # ---------------------------------------------------------------- 3
    section(out, "3. Read the chunks. Every time.")
    out.append(
        "Five questions come out of §2 as failures or adjudications. I opened the retrieved "
        "chunks for every one. **Not a single one was the failure the metrics said it was.**\n\n"
        "| id | RAGAS says | actually |\n|---|---|---|\n"
        "| `p07` | worst question in the set (faith 0.00, relevancy 0.00) | **the system "
        "behaved perfectly.** `TreeSet.html` was retrieved, but the chunk was the "
        "`synchronizedSortedSet` boilerplate, not the constructor prose that says `add` "
        "throws `ClassCastException`. The gold sentence exists in the corpus and was never "
        "pulled. The model said `insufficient context` — the correct response to missing "
        "evidence — and RAGAS scored a refusal as a hallucination. A **retrieval** bug, "
        "punished as a generation bug. |\n"
        "| `s14` | faithfulness **0.00** — total hallucination | the answer is `It returns "
        "null`, and chunk 0 reads *\"or returns null if this deque is empty\"*. It is "
        "verbatim correct. RAGAS decomposes an answer into claims and verifies each; a "
        "three-word answer yields no claim to verify and the score collapses to zero. A "
        "**judge artifact**. Note it flipped 1.00 → 0.00 purely from reranking (§4) — the "
        "same right answer, scored differently on two runs. |\n"
        "| `p10` | recall **0.00** — retriever failed | the model answered with "
        "`String.join`, cited `java.lang.String.html`, and was faithful (1.00) and right. "
        "The eval set had written down `Collectors.joining` as the only acceptable source. "
        "Two valid answers, one gold label. An **eval-set** bug — no code change would fix "
        "it, and 'fixing' the retriever to chase it would make the system worse. **FIXED:** "
        "gold widened to accept both; recall 0.00 → **1.00**, precision 0.00 → **1.00**. |\n"
        "| `p06` | retriever failure (recall 0.00) | **the only genuine model error in the "
        "set,** and it is a good one. The gold sentence (*\"Returns a fixed-size list backed "
        "by the specified array\"*) was NOT retrieved — but the chunk that *was* retrieved "
        "contains both *\"a convenient way to create a **fixed-size** list\"* and *\"The list "
        "returned by this method **is modifiable**\"*. The model read the second and answered "
        "\"Yes, you can add elements.\" You cannot: `add()` throws "
        "`UnsupportedOperationException`. The JavaDoc's own wording is the trap — *modifiable* "
        "means `set()` works, not `add()` — and the model walked straight into it. Half "
        "retrieval near-miss, half genuine misreading. |\n"
        "| `p01` | faith 1.00, recall 1.00 — looks **fine**, never flagged before | asked *\"how "
        "do I read a file one line at a time?\"*, the model answered `Files.lines()`. Correct, "
        "idiomatic, well-cited. The gold label says `BufferedReader`. Same story as `p10`: a "
        "**second** eval-set bug, and the old file-level metric could not see it because "
        "`BufferedReader.html` *was* retrieved — just not the sentence. Passage-level hit is "
        "what surfaced it. **FIXED:** gold widened to accept both; under the full system its "
        "precision went 0.42 → **1.00**. |\n"
    )
    out.append(
        "> The scoreboard said: 2 retriever failures, 2 LLM failures. The chunks said: "
        "**1 retrieval miss, 1 genuine model error, 2 broken gold labels, 1 judge artifact** — "
        "and a correct refusal filed as the worst answer in the run. Every single verdict the "
        "metrics handed down named the wrong culprit, and the passage-level check found a sixth "
        "problem (`p01`) that scored 1.00/1.00 and looked perfect.\n>\n"
        "> This is not an argument against RAGAS. It is an argument against reading RAGAS "
        "without opening the chunks underneath it. **The single genuine model bug in 55 "
        "questions is `p06`, and no metric in the suite pointed at it.**"
    )
    out.append(
        "\n**What was actually changed.** Only the eval set. `p01` and `p10` now carry "
        "`alt_golds` (see `evaluation/gold.py`), the retrieval metrics accept any gold, and "
        "the two questions were re-judged on the only two metrics a gold label can move — "
        "`context_precision` and `context_recall`. `faithfulness` and `answer_relevancy` never "
        "see the reference and were left untouched on purpose: re-rolling 43 unchanged "
        "questions through a judge that scores the same answer 1.00 one run and 0.00 the next "
        "would have buried the fix under fresh noise.\n\n"
        "**The retriever was never touched, and the numbers went up anyway** — because two of "
        "the four failures were never in the retriever. That is the whole argument for reading "
        "the chunks before you start optimising:\n\n"
        "```\n"
        "                     context_precision      context_recall\n"
        "dense_only            0.688  ->  0.694     0.904  ->  0.926\n"
        "sparse_only           0.681  ->  0.700     0.815  ->  0.859\n"
        "hybrid                0.743  ->  0.761     0.889  ->  0.911\n"
        "hybrid_rerank         0.746  ->  0.781     0.944  ->  0.967\n"
        "```\n"
        "The ranking is unchanged and `hybrid_rerank` still wins every column, so the Day 8 "
        "conclusion holds. But ~3.5 points of the full system's context_precision was never a "
        "system deficiency at all — it was a bad ruler."
    )

    # ---------------------------------------------------------------- 4
    section(out, f"4. Worst {worst} questions in the full system")
    sub["mean_score"] = sub[METRICS].mean(axis=1)
    cols = ["id", "query_type", "mean_score"] + METRICS + ["file_hit", "passage_hit"]
    out.append(
        "```\n"
        + sub.nsmallest(worst, "mean_score")[cols].round(3).to_string(index=False)
        + "\n```"
    )

    # ---------------------------------------------------------------- 5
    section(out, "5. What reranking changed, question by question")
    a = scored[scored["mode"] == "hybrid"].set_index("id")
    b = scored[scored["mode"] == FULL_SYSTEM].set_index("id")
    if not a.empty and not b.empty:
        delta = (b[METRICS] - a[METRICS]).round(3)
        delta["passage_gained"] = (~a["passage_hit"].astype(bool)) & b["passage_hit"].astype(bool)
        delta["passage_lost"] = a["passage_hit"].astype(bool) & (~b["passage_hit"].astype(bool))
        moved = delta[(delta[METRICS].abs() > 0.01).any(axis=1)]
        out.append(
            f"Reranking gained the gold **passage** on "
            f"**{int(delta['passage_gained'].sum())}** questions and lost it on "
            f"**{int(delta['passage_lost'].sum())}**.\n"
        )
        out.append(
            "All four modes cut to the same top 5, but the un-reranked ones take the FIRST 5 "
            "of the 20 candidates while the reranker RE-PICKS which 5 survive. So it does not "
            "merely reorder the window — it changes what is in it. That is why its gains land "
            "on recall-flavoured metrics, not on context_precision as I predicted on Day 4.\n\n"
            "Read the `faithfulness` column with suspicion, though: `s14` and `p07` swing a "
            "full **-1.000** here, and §3 shows both swings are judge noise on a correct "
            "answer and a correct refusal. Some of the cross-encoder's apparent effect on "
            "the headline table is the judge changing its mind, not the system changing its "
            "behaviour.\n"
        )
        out.append("```\n" + moved.to_string() + "\n```")

    # ---------------------------------------------------------------- 6
    section(out, "6. Every metric, every config")
    grid = (
        scored.groupby("mode")[METRICS]
        .mean()
        .join(ans.groupby("mode")[["file_hit", "passage_hit"]].mean())
        .join(
            runs[~runs["answerable"]].groupby("mode")["refused"].mean().rename("abstention")
        )
        .round(3)
        .reindex(MODES)
    )
    out.append("```\n" + grid.to_string() + "\n```")
    out.append(
        "\n`hybrid_rerank` is still the best configuration on every column, so the headline "
        "conclusion survives. What does not survive is the reason I believed it: `file_hit` "
        "1.000 was never the win it looked like, and roughly half the faithfulness gap "
        "between configs is judge noise on four questions."
    )

    report = "\n".join(out)
    REPORT_PATH.write_text(report, encoding="utf-8")
    # The Windows console is cp1252 and dies on the arrows and em-dashes that the
    # report is full of. The file is UTF-8 and correct; only stdout needs babying.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)
    print(f"\n\nsaved -> {REPORT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default=FULL_SYSTEM, choices=MODES)
    p.add_argument("--worst", type=int, default=8)
    a = p.parse_args()
    main(a.mode, a.worst)
