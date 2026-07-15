"""
ablation_runner.py  —  Days 8-9: measure the system, one component at a time
============================================================================

An ABLATION means: remove one piece, re-measure, and see what breaks. It is how
you prove each component earns its place instead of assuming it does. We run the
same 55 questions through four configurations:

    dense_only      semantic search only
    sparse_only     BM25 keyword search only
    hybrid          RRF fusion of both
    hybrid_rerank   RRF fusion + cross-encoder rerank   (the full system)

and score each with RAGAS:

    faithfulness       is every claim in the answer supported by the chunks?
                       -> catches HALLUCINATION
    answer_relevancy   does the answer actually address the question?
    context_precision  are the retrieved chunks relevant, best ones first?
                       -> this is what RERANKING should improve
    context_recall     was the needed evidence retrieved at all?
                       -> this is what RETRIEVAL STRATEGY should improve

The 10 unanswerable questions are scored SEPARATELY, with `refusal_rate`. They
have no ground-truth answer and no source passage, so context_recall and
context_precision are mathematically undefined for them and faithfulness on the
string "insufficient context" is meaningless. What they actually test is whether
the system refuses instead of leaking Gemini's pretrained knowledge -- so we
measure exactly that, with a free, local string check and zero API calls.

--------------------------------------------------------------------------
TWO STAGES, because the free tier is the real constraint here
--------------------------------------------------------------------------
Stage 1 (`--stage 1`) runs the pipeline: 55 questions x 4 configs = 220 LLM
calls. Results are CACHED to results/raw_runs.json, keyed by mode+question id.
Re-running never re-pays for a generation already done.

Stage 2 (`--stage 2`) runs the RAGAS judge: ~45 answerable x 4 configs x 4
metrics, and faithfulness alone costs two calls per sample (extract the claims,
then verify each). That is several hundred more calls. It is saved PER MODE, so
if the daily quota (1000 requests/day) runs out mid-way, tomorrow's run picks up
exactly where it stopped instead of starting over.

Everything is rate-limited to --rpm requests/minute (default 15).

    python -m evaluation.ablation_runner --stage 1
    python -m evaluation.ablation_runner --stage 2
    python -m evaluation.ablation_runner --stage table
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from generation.pipeline import answer, MODES
from generation.generator import GEN_MODEL, INSUFFICIENT
from ingestion.indexer import EMBED_MODEL
from evaluation.gold import references

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "eval" / "eval_set.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_PATH = RESULTS_DIR / "raw_runs.json"
TABLE_PATH = RESULTS_DIR / "ablation_table.csv"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# The RAGAS judge. Deliberately NOT the model that wrote the answers (GEN_MODEL):
# a model shown its own output over-rates it, and the free tier's 500-requests/day
# cap is per model, so a second model is also a second quota pool.
# gemma-4-31b-it: 1500 req/day, 15 rpm. Verified to survive ragas' structured output.
JUDGE_MODEL = "gemma-4-31b-it"


# ---------------------------------------------------------------- helpers

def load_eval_set() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def load_raw() -> dict:
    if RAW_PATH.exists():
        return json.loads(RAW_PATH.read_text(encoding="utf-8"))
    return {}


def save_raw(raw: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    RAW_PATH.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8")


def percentile(values: list[float], p: float) -> float:
    """P50/P95 without numpy. Linear interpolation between the two nearest ranks."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def refused(text: str) -> bool:
    return text.strip().lower().startswith(INSUFFICIENT)


# The ONLY two metrics that read `reference`. faithfulness compares the answer to
# the retrieved chunks; answer_relevancy compares the answer to the question.
# Neither one can move when a gold label changes -- which is what makes the
# targeted `--rescore` below exactly equivalent to a full re-run, not an
# approximation of one.
REF_DEPENDENT = ["context_precision", "context_recall"]


def metric_col(df, name: str) -> str | None:
    """RAGAS names columns after the metric CLASS, so context_precision arrives as
    'llm_context_precision_with_reference'. Match on substring."""
    return next((c for c in df.columns if name in c), None)


def collapse_multi_reference(df):
    """One row per question, taking the BEST accepted gold.

    A question with alternate golds was scored once per gold. The question we are
    asking is "did the system find A right answer?", not "did it find the first one
    I happened to write down" -- so the reference-dependent metrics take the max
    across golds. The other metrics are identical across the duplicate rows (they
    never saw the reference), so first() and max() agree and it does not matter.
    """
    if not df["id"].duplicated().any():
        return df
    agg = {c: "max" if any(m in c for m in REF_DEPENDENT) else "first"
           for c in df.columns if c != "id"}
    return df.groupby("id", sort=False).agg(agg).reset_index()


# ---------------------------------------------------------------- stage 1

def stage_run(rpm: int, modes: list[str], limit: int = 0) -> None:
    """Run the pipeline for every (mode, question) pair, caching as we go."""
    examples = load_eval_set()
    if limit:  # smoke test: prove the plumbing works before spending 220 calls
        examples = examples[:limit]
    raw = load_raw()
    min_gap = 60.0 / rpm  # seconds between LLM calls, to respect the free tier

    todo = [(m, ex) for m in modes for ex in examples if f"{m}::{ex['id']}" not in raw]
    print(f"stage 1: {len(todo)} runs to do "
          f"({len(raw)} already cached), pacing at {rpm} req/min\n")

    for i, (mode, ex) in enumerate(todo, start=1):
        key = f"{mode}::{ex['id']}"
        start = time.perf_counter()

        for attempt in range(5):
            try:
                result = answer(ex["question"], mode=mode)
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"  [{key}] error: {str(e).splitlines()[0][:70]} — retry in {wait}s")
                time.sleep(wait)
        else:
            print(f"  [{key}] FAILED after retries; skipping")
            continue

        raw[key] = {
            "id": ex["id"],
            "mode": mode,
            "query_type": ex["query_type"],
            "question": ex["question"],
            "answer": result["answer"],
            "contexts": [c.page_content for c in result["chunks"]],
            "sources": result["sources"],
            "reference": ex["ground_truth_answer"],
            "reference_passage": ex["source_passage"],
            "timings": result["timings"],
        }
        save_raw(raw)  # save after EVERY run: a crash costs one call, not 220

        elapsed = time.perf_counter() - start
        print(f"  [{i}/{len(todo)}] {key:34s} {result['timings']['total_ms']:6.0f} ms")
        if i < len(todo):
            time.sleep(max(0.0, min_gap - elapsed))

    print(f"\nstage 1 done — {len(raw)} runs cached in {RAW_PATH.name}")


# ---------------------------------------------------------------- stage 2

def stage_score(rpm: int, modes: list[str]) -> None:
    """Score the answerable questions of each mode with RAGAS, one mode at a time."""
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    # RAGAS 0.4 keeps the classic metric classes in private modules.
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall

    raw = load_raw()
    if not raw:
        raise SystemExit("no cached runs — run `--stage 1` first")

    gold_by_id = {e["id"]: e for e in load_eval_set()}
    api_key = os.getenv("GOOGLE_API_KEY")

    def make_judge(model: str):
        # temperature=0 so the same answer always gets the same score: an
        # evaluation that moves on its own is not an evaluation.
        # The rate limiter is what keeps hundreds of judge calls inside the free tier.
        return LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=0.0,
            rate_limiter=InMemoryRateLimiter(
                requests_per_second=rpm / 60.0, max_bucket_size=1),
        ))

    gemma = make_judge(JUDGE_MODEL)   # independent judge, 1500 calls/day
    flash = make_judge(GEN_MODEL)     # our own generator,  500 calls/day
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)
    )

    # WHICH MODEL JUDGES WHICH METRIC -- two independent constraints decide this.
    #
    # 1. BIAS. A model shown its own output over-rates it. faithfulness and
    #    answer_relevancy both take our `response` as input, so if GEN_MODEL judged
    #    them it would be grading its own homework. They go to gemma.
    #    context_precision and context_recall never see the response at all -- they
    #    compare the retrieved chunks against the reference passage -- so GEN_MODEL
    #    can judge those with no self-bias.
    #
    # 2. QUOTA. The free tier is 500 requests/day PER MODEL, so two models = two
    #    separate pools. Per sample: faithfulness 2, answer_relevancy 1,
    #    context_precision 1-per-chunk = 5, context_recall 1.  x45 answerable x4
    #    configs  ->  gemma 1440/1500, flash-lite 180/500.  It fits.
    #
    # A metric must keep the SAME judge across all four configs, or the ablation
    # would be comparing configs and judges at once and prove nothing. So metrics
    # are assigned to models whole; never split one metric across two.
    metrics = [
        Faithfulness(llm=gemma),
        # strictness=1 is REQUIRED here, not a tuning choice. answer_relevancy works
        # by generating N candidate questions from the answer and measuring how close
        # they are to the real question -- and it asks for all N in one call (n=N).
        # Gemini rejects n>1 ("Multiple candidates is not enabled for this model"),
        # so the metric silently returns NaN at the default strictness=3.
        # The cost: one sampled question instead of an average of three, so this
        # metric is noisier than a reference RAGAS run would be. Worth stating in
        # the README rather than quietly shipping a number that isn't comparable.
        ResponseRelevancy(llm=gemma, embeddings=judge_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=gemma),
        LLMContextRecall(llm=flash),
    ]
    # Each ChatGoogleGenerativeAI holds its OWN rate limiter and shares it across
    # threads, so workers>1 does not break the --rpm budget: it just stops us idling.
    # The floor is arithmetic -- 1440 gemma calls / 15 rpm = 96 min no matter what --
    # but at max_workers=1 we measured 26 s per job, which would take 5 h. Overlapping
    # the two models (they have separate rpm limits) gets us close to the floor.
    run_config = RunConfig(max_workers=4, timeout=180, max_retries=5)

    # State the bill before running it up. Per sample: faithfulness 2,
    # answer_relevancy 1, context_precision 1-per-retrieved-chunk, context_recall 1.
    todo = [m for m in modes if not (RESULTS_DIR / f"ragas_{m}.csv").exists()]
    n = sum(len([r for r in raw.values()
                 if r["mode"] == m and r["query_type"] != "unanswerable"]) for m in todo)
    k = max((len(r["contexts"]) for r in raw.values()), default=5)
    print(f"judge={JUDGE_MODEL} (faithfulness, answer_relevancy, context_precision)")
    print(f"judge={GEN_MODEL} (context_recall)")
    print(f"{len(todo)} mode(s) to score, {n} samples, {k} chunks each")
    print(f"  {JUDGE_MODEL:20s} ~{n * (2 + 1 + k):5d} calls  (cap 1500/day)")
    print(f"  {GEN_MODEL:20s} ~{n * 1:5d} calls  (cap  500/day)\n")

    for mode in modes:
        out_path = RESULTS_DIR / f"ragas_{mode}.csv"
        if out_path.exists():
            print(f"{mode}: already scored ({out_path.name}) — skipping")
            continue

        rows = [r for r in raw.values()
                if r["mode"] == mode and r["query_type"] != "unanswerable"]
        if not rows:
            print(f"{mode}: no cached answerable runs — run stage 1 first")
            continue

        print(f"\n{mode}: scoring {len(rows)} answerable questions with RAGAS...")

        # The reference comes from the EVAL SET, not from r["reference"].
        # Stage 1 denormalized the gold answer into raw_runs.json, which meant that
        # fixing a bad gold label in eval_set.json changed NOTHING here -- the judge
        # kept scoring against the stale copy baked in at generation time. That is a
        # silent trap: you edit the ruler, re-run, and get the old numbers back.
        # Read the ruler where the ruler lives.
        #
        # A question may have several accepted golds (see evaluation/gold.py), so we
        # emit one row PER reference and take the best one afterwards. Only the two
        # reference-dependent metrics can differ between them; faithfulness and
        # answer_relevancy never see the reference at all.
        expanded, owners = [], []
        for r in rows:
            for ref in references(gold_by_id[r["id"]]):
                expanded.append({
                    "user_input": r["question"],
                    "response": r["answer"],
                    "retrieved_contexts": r["contexts"],
                    "reference": ref,
                })
                owners.append(r)

        result = evaluate(
            dataset=EvaluationDataset.from_list(expanded),
            metrics=metrics,
            llm=gemma,               # only a fallback; every metric sets its own
            embeddings=judge_embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )

        df = result.to_pandas()
        df.insert(0, "id", [r["id"] for r in owners])
        df.insert(1, "query_type", [r["query_type"] for r in owners])
        df = collapse_multi_reference(df)
        RESULTS_DIR.mkdir(exist_ok=True)

        # THE POISON GUARD. When a judge call fails -- quota, timeout, bad JSON --
        # ragas does not crash. It writes NaN for that one cell, finishes the run,
        # and exits 0 saying "saved". A mode that died 60% of the way through the
        # quota therefore leaves behind a CSV that LOOKS complete and has entirely
        # plausible means. Worse, stage_score skips any mode whose CSV exists, so
        # tomorrow's run would never re-score it and the ablation table would be
        # built on a silently biased subset. That nearly happened on 2026-07-13.
        # So: a CSV with ANY NaN is not a result, it is a partial. Park it under a
        # name stage_score does not skip, and make the failure loud.
        cols = [c for c in df.columns
                if any(m in c for m in METRIC_NAMES) and c not in ("id", "query_type")]
        n_nan = int(df[cols].isna().sum().sum())
        if n_nan:
            partial = RESULTS_DIR / f"ragas_{mode}.PARTIAL.csv"
            df.to_csv(partial, index=False, encoding="utf-8")
            per_col = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
            print(f"{mode}: INCOMPLETE — {n_nan} unscored cells {per_col}")
            print(f"{mode}: parked -> {partial.name} (NOT counted; re-run to redo this mode)")
            print(f"{mode}: stopping. Fix the cause (quota?) before scoring more modes.")
            return
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"{mode}: saved -> {out_path.name}  ({len(df)} rows, 0 NaN)")


# ---------------------------------------------------------------- rescore

def stage_rescore(rpm: int, modes: list[str], ids: list[str]) -> None:
    """Re-judge specific questions after their GOLD LABEL changed, and patch the CSVs.

    Why this exists instead of just deleting the CSVs and re-running stage 2:

    Changing a gold label can only move `context_precision` and `context_recall`.
    Those are the only two metrics that are shown the reference. `faithfulness`
    (answer vs chunks) and `answer_relevancy` (answer vs question) never see it and
    are mathematically incapable of moving. So re-scoring two questions on two
    metrics gives BIT-FOR-BIT the result a full re-score would give, for ~96 judge
    calls instead of ~1600.

    That is not a shortcut taken to save money, though it does. A full re-run would
    also re-roll every OTHER question through the judge, and this judge is not
    perfectly stable -- §3 of the failure analysis caught it scoring the same correct
    answer 1.00 and 0.00 on two different runs. Re-rolling 43 questions that did not
    change would inject fresh judge noise into every one of them and make the before/
    after comparison meaningless: you could no longer tell the gold-label fix from
    the drift. Touching only what changed is the more correct experiment, not just
    the cheaper one.
    """
    import pandas as pd
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall

    raw = load_raw()
    gold_by_id = {e["id"]: e for e in load_eval_set()}
    api_key = os.getenv("GOOGLE_API_KEY")

    def make_judge(model: str):
        return LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=0.0,
            rate_limiter=InMemoryRateLimiter(
                requests_per_second=rpm / 60.0, max_bucket_size=1),
        ))

    # Same judge-to-metric assignment as stage 2. A metric that changed judges
    # between the original run and this patch would make the columns incomparable.
    metrics = [
        LLMContextPrecisionWithReference(llm=make_judge(JUDGE_MODEL)),
        LLMContextRecall(llm=make_judge(GEN_MODEL)),
    ]
    run_config = RunConfig(max_workers=4, timeout=180, max_retries=5)

    for mode in modes:
        path = RESULTS_DIR / f"ragas_{mode}.csv"
        if not path.exists():
            print(f"{mode}: no {path.name} to patch — skipping")
            continue

        df = pd.read_csv(path)
        rows, owners = [], []
        for qid in ids:
            key = f"{mode}::{qid}"
            if key not in raw or qid not in gold_by_id:
                print(f"  {key}: not cached — skipping")
                continue
            r = raw[key]
            for ref in references(gold_by_id[qid]):
                rows.append({
                    "user_input": r["question"],
                    "response": r["answer"],
                    "retrieved_contexts": r["contexts"],
                    "reference": ref,
                })
                owners.append(qid)

        if not rows:
            continue

        print(f"\n{mode}: re-judging {len(rows)} (question, gold) pairs "
              f"for {len(set(owners))} question(s) on {REF_DEPENDENT}...")
        result = evaluate(
            dataset=EvaluationDataset.from_list(rows),
            metrics=metrics,
            run_config=run_config,
            raise_exceptions=False,
        )
        new = result.to_pandas()
        new.insert(0, "id", owners)

        cols = {m: metric_col(new, m) for m in REF_DEPENDENT}
        if any(c is None for c in cols.values()):
            raise SystemExit(f"{mode}: judge returned no column for {cols} — aborting")

        # Same poison guard as stage 2: a NaN means a judge call failed (quota,
        # timeout, bad JSON). Writing it would corrupt a good CSV with a blank cell
        # and the mean would quietly shift. Refuse to write.
        n_nan = int(new[list(cols.values())].isna().sum().sum())
        if n_nan:
            print(f"{mode}: INCOMPLETE — {n_nan} unscored cell(s). "
                  f"CSV left untouched. Fix the cause (quota?) and re-run.")
            return

        best = collapse_multi_reference(new).set_index("id")
        for qid in best.index:
            target = df["id"] == qid
            for m, col in cols.items():
                old_val = float(df.loc[target, metric_col(df, m)].iloc[0])
                new_val = float(best.loc[qid, col])
                df.loc[target, metric_col(df, m)] = new_val
                # ASCII only: this prints to a cp1252 console, and a UnicodeEncodeError
                # here would abort AFTER the judge calls were paid for but BEFORE the
                # CSV was written -- burning quota for nothing. Learned the hard way.
                arrow = "->" if abs(new_val - old_val) > 1e-9 else " ="
                print(f"  {qid:4s} {m:18s} {old_val:.3f} {arrow} {new_val:.3f}")

        df.to_csv(path, index=False, encoding="utf-8")
        print(f"{mode}: patched -> {path.name}")


# ---------------------------------------------------------------- table

def stage_table(modes: list[str]) -> None:
    """Combine the RAGAS scores, the refusal rate and the latencies into one CSV."""
    import pandas as pd

    raw = load_raw()
    lines = []

    for mode in modes:
        runs = [r for r in raw.values() if r["mode"] == mode]
        if not runs:
            continue

        scores: dict[str, float] = {}
        csv_path = RESULTS_DIR / f"ragas_{mode}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for m in METRIC_NAMES:
                # RAGAS names columns after the metric CLASS, so context_precision
                # arrives as 'llm_context_precision_with_reference'. Match on substring.
                col = next((c for c in df.columns if m in c), None)
                scores[m] = float(df[col].mean()) if col else float("nan")

        # Refusal rate on the unanswerable questions: no LLM judge needed.
        unanswerable = [r for r in runs if r["query_type"] == "unanswerable"]
        refusal_rate = (
            sum(refused(r["answer"]) for r in unanswerable) / len(unanswerable)
            if unanswerable else float("nan")
        )

        latencies = [r["timings"]["total_ms"] for r in runs]

        lines.append({
            "config": mode,
            **{m: round(scores.get(m, float("nan")), 4) for m in METRIC_NAMES},
            "refusal_rate_unanswerable": round(refusal_rate, 4),
            "latency_p50_ms": round(percentile(latencies, 0.50)),
            "latency_p95_ms": round(percentile(latencies, 0.95)),
            "n_answerable": len(runs) - len(unanswerable),
            "n_unanswerable": len(unanswerable),
        })

    RESULTS_DIR.mkdir(exist_ok=True)
    table = pd.DataFrame(lines)
    table.to_csv(TABLE_PATH, index=False, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nsaved -> {TABLE_PATH}")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "rescore", "table", "all"], default="all")
    parser.add_argument("--rpm", type=int, default=15, help="requests/minute budget")
    parser.add_argument("--modes", nargs="*", default=MODES)
    parser.add_argument("--limit", type=int, default=0, help="only first N questions (smoke test)")
    parser.add_argument("--ids", nargs="*", default=[],
                        help="question ids to re-judge with --stage rescore "
                             "(use after fixing a gold label in eval_set.json)")
    args = parser.parse_args()

    if args.stage in ("1", "all"):
        stage_run(args.rpm, args.modes, args.limit)
    if args.stage in ("2", "all"):
        stage_score(args.rpm, args.modes)
    if args.stage == "rescore":
        if not args.ids:
            raise SystemExit("--stage rescore needs --ids (e.g. --ids p01 p10)")
        stage_rescore(args.rpm, args.modes, args.ids)
    if args.stage in ("rescore", "table", "all"):
        stage_table(args.modes)
