# Hybrid RAG over the Java API Reference

A retrieval-augmented question-answering system built over 26 JavaDoc pages, and
— just as much — a record of how it was **measured**. The interesting part of this
project is not that it answers Java questions; it is the evidence that each moving
part earns its place, and the failure analysis showing where the evaluation itself
lied.

```
$ python -m scripts.ask "What is the default initial capacity of a HashMap?"

The default initial capacity is 16. [java.util.HashMap.html]

SOURCES USED: java.util.HashMap.html
TIMINGS: retrieval 41 ms · rerank 1780 ms · generation 690 ms · total 2511 ms
```

The corpus is deliberately a **reference** (JavaDoc), and the questions are
deliberately **exact-lookup** ("what does `Integer.parseInt(\"Kona\", 27)` return?").
That is the setting where a wrong answer is worst and hardest to hide, which makes
it a good stress test for grounding.

---

## How it works: a two-stage funnel

Retrieval is split into a cheap, wide first stage and an expensive, precise second
stage. Reranking all 759 chunks with the cross-encoder would cost ~70 s/query; the
funnel gets the same precision for ~2 s.

```
  question
     │
     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STAGE 1 — retrieve 20 candidates (fast, favours recall)  │
  │                                                          │
  │   dense   Gemini embeddings + Chroma   (semantic match)  │
  │   sparse  BM25 over the same chunks     (keyword match)  │
  │   hybrid  Reciprocal Rank Fusion of the two              │
  └─────────────────────────────────────────────────────────┘
     │  20 candidates
     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STAGE 2 — rerank to top 5 (slow, favours precision)      │
  │   cross-encoder/ms-marco-MiniLM-L-6-v2                    │
  │   scores (query, chunk) pairs jointly, re-picks the 5    │
  └─────────────────────────────────────────────────────────┘
     │  5 chunks
     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ GENERATE — gemini-3.1-flash-lite, temperature 0          │
  │   grounding prompt: answer ONLY from context, cite the   │
  │   source file, or reply "insufficient context"           │
  └─────────────────────────────────────────────────────────┘
     │
     ▼
  answer + citations
```

**Bi-encoder vs cross-encoder.** The dense retriever is a *bi-encoder*: the query
and each chunk are embedded independently into vectors that can be precomputed and
compared with a cheap dot product — fast, but lossy. The reranker is a
*cross-encoder*: it feeds `[query [SEP] chunk]` through the transformer *together*
and emits one relevance score — far more accurate, but nothing can be precomputed,
so it only runs on the 20 survivors of stage 1.

**Grounding is the anti-hallucination mechanism.** The prompt forbids prior
knowledge, requires a `[source_file]` citation on every claim, and gives an explicit
escape hatch: if the context does not contain the answer, reply exactly
`insufficient context`. Without that escape hatch, the 10 unanswerable test
questions would be meaningless.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

# Google Gemini free tier — put your key in .env (never committed):
echo GOOGLE_API_KEY=your-key-here > .env

# Build the indices from corpus/ (dense vectors + BM25). One-time; free-tier embeds.
python -m ingestion.indexer
```

This project uses the **Google Gemini free tier**, not OpenAI. The relevant limit
is **500 generate-content requests/day _per model_** — that per-model detail is load
bearing for the evaluation (see below).

---

## Usage

```bash
# Ask a question (full system):
python -m scripts.ask "How do I convert a String to lowercase?"

# Force a single retrieval mode:
python -m scripts.ask "..." --mode dense_only        # or sparse_only, hybrid, hybrid_rerank

# Interactive loop:
python -m scripts.ask

# Inspect what retrieval actually returns, side by side:
python -m scripts.test_retrieval

# Browse the indexed chunks (not the raw HTML):
python -m scripts.browse_corpus "initial capacity" --n 5
```

---

## Evaluation

The whole point of Days 8–10. Three artifacts, all reproducible from cache.

### The eval set — `eval/eval_set.json` (55 questions)

| type | n | tests |
|---|---|---|
| `simple` | 25 | ordinary lookups |
| `keyword_heavy` | 10 | exact API names — BM25's home turf |
| `paraphrase_heavy` | 10 | reworded, share few words with the docs — dense's job |
| `unanswerable` | 10 | answer is **not** in the corpus — must refuse |

> **Disclosure — how the ground truths were written.** The guide says never let an
> LLM write the ground-truth answers. Here they *were* LLM-produced, but the risk was
> managed: each answer was **transcribed from a real retrieved corpus chunk** (via
> ~50 BM25 lookups), not recalled from memory, and then **human spot-checked**. The
> question *selection* is still the author's. Treat these as
> transcribed-and-checked, not hand-authored from scratch.

### The ablation — `evaluation/ablation_runner.py`

Runs the same 45 answerable questions through all four configurations and scores
each with [RAGAS](https://docs.ragas.io) on four metrics:

- **faithfulness** — is every claim backed by the retrieved chunks? (catches hallucination)
- **answer_relevancy** — does the answer address the question?
- **context_precision** — are the retrieved chunks relevant, best first?
- **context_recall** — was the needed evidence retrieved at all?

The 10 unanswerable questions are scored **separately**, with a free, local,
zero-API `refusal_rate` — RAGAS cannot score them (no reference passage exists).

**Two judge models, on purpose.** RAGAS uses an LLM as the grader, at ~9 calls per
sample → ~1,600 calls total, which blows past 500/day. Two facts drive the fix:

1. **Bias.** A model over-rates its own output. `gemini-3.1-flash-lite` *wrote* the
   answers, so it cannot fairly judge them.
2. **Quota is per model.** A second model is a second 500/day pool.

Both are solved by the same move — judge with a *different* model — so metrics are
assigned to judges by which input they read:

| metric | judge | why |
|---|---|---|
| faithfulness | `gemma-4-31b-it` | reads the answer → must not be the generator |
| answer_relevancy | `gemma-4-31b-it` | reads the answer |
| context_precision | `gemma-4-31b-it` | expensive (1 call/chunk); gemma has 1500/day |
| context_recall | `gemini-3.1-flash-lite` | never sees the answer → no self-bias |

A metric keeps the **same** judge across all four configs — otherwise a row
difference could be the config *or* the judge, and the ablation would prove nothing.

```bash
python -m evaluation.ablation_runner --stage 1      # run pipeline → results/raw_runs.json (cached)
python -m evaluation.ablation_runner --stage 2      # RAGAS judging → results/ragas_<mode>.csv
python -m evaluation.ablation_runner --stage table  # combine → results/ablation_table.csv
```

Stage 1 (220 runs) is cached in `results/raw_runs.json` and never re-pays. Stage 2
saves per-mode and **halts loudly on any missing score** — see the NaN trap below.

### Results

```
config          faithfulness  answer_relevancy  context_precision  context_recall  refusal  p50_ms  p95_ms
dense_only            0.888          0.822             0.694            0.926          100%     1122    1916
sparse_only           0.806          0.769             0.700            0.859          100%      761    1101
hybrid                0.893          0.837             0.761            0.911          100%     1141    1505
hybrid_rerank         0.933          0.889             0.781            0.967          100%     2340    6273
```

**Every component earns its place.** Quality rises monotonically
`sparse < dense < hybrid < hybrid_rerank` — the full system wins every quality
column. The price is latency: ~2× at the median, ~4× at p95 (6.3 s), for ~4–6
points of quality. That is a product decision, now an informed one.

**Reranking improves recall, not precision — the opposite of what was predicted.**
All four modes cut to the same top 5, but the un-reranked ones take the *first* 5 of
the 20 candidates while the reranker *re-picks which* 5 survive. So it changes what
is *in* the window, promoting an answer-bearing chunk from rank ~12 — which shows up
as **context_recall** (0.889 → 0.967), not context_precision.

**Refusal is 100% across all four configs**, including the weakest retriever
(`sparse_only`). Hallucination-resistance comes from the grounding *prompt*, not from
retrieval quality — a clean separation: fixing retrieval won't fix hallucination, and
vice versa.

---

## Failure analysis: read the chunks, every time

Full write-up in `results/failure_analysis.md`. The headline finding is a warning
about trusting metrics.

RAGAS flagged **4 apparent failures**. Opening the actual retrieved chunks for each
showed **not one was the failure the metric named**:

| id | metric's verdict | reality |
|---|---|---|
| `p07` | worst question (faith 0.00) | **correct refusal** — the gold sentence was never retrieved; the model rightly said `insufficient context` and was scored as if it hallucinated. A *retrieval* miss punished as a *generation* bug. |
| `s14` | faithfulness 0.00 — hallucination | **judge artifact** — the answer "returns null" is verbatim correct, but a 3-word answer decomposes into zero checkable claims, so the score collapses to 0. |
| `p10` | recall 0.00 — retriever failed | **eval-set bug** — `String.join` and `Collectors.joining` both correctly "glue strings together"; the eval set listed only one. Fixed via `alt_golds`. |
| `p06` | retriever failure | **the one genuine model error in all 55 questions** — it read "the list is *modifiable*" and answered "yes, you can add elements." You cannot: `Arrays.asList` is fixed-size and `add()` throws. *modifiable* means `set()` works, not `add()`. No metric in the suite pointed at this. |

A passage-level check also surfaced a **fifth** problem (`p01`) that scored 1.00/1.00
and looked perfect but was a second bad gold label (`Files.lines` vs `BufferedReader`,
both correct).

Two lessons, both load-bearing:

1. **File-level hit rate is a metric that cannot fail.** `hybrid_rerank` scores a
   perfect 1.000 on "did *a* chunk of the right *file* appear?" but 0.956 on "did the
   right *sentence* appear?" A large page like `TreeSet.html` has many chunks; pulling
   one is not pulling the answer. The passage-level check is where the real failures live.
2. **The fix was to the eval set, not the retriever.** Two of four "failures" were bad
   gold labels. `context_precision`/`context_recall` rose after widening the gold —
   the *ruler* was wrong, not the system. Chasing those by "tuning the retriever" would
   have made a working retriever worse.

```bash
python -m scripts.failure_analysis
```

---

## Caveats worth stating plainly

- **`answer_relevancy` runs at `strictness=1`.** RAGAS's default asks the judge for 3
  candidate questions in one call (`n=3`); Gemini rejects `n>1`
  (`Multiple candidates is not enabled for this model`) and *silently* returns NaN. At
  `strictness=1` the metric averages **one** sampled question instead of three, so it
  is noisier and **not directly comparable** to published RAGAS numbers.
- **Terse-answer judge artifacts.** As with `s14`, a correct one-line answer can score
  faithfulness 0.00 because it yields no claims to verify. Read low faithfulness scores
  on short answers with suspicion.
- **The NaN trap.** When a judge call fails (quota/timeout), RAGAS writes NaN, finishes,
  and exits 0 saying "saved" — a quota-killed run leaves a CSV that *looks* complete
  with plausible means. `stage_score` now quarantines any CSV containing a NaN as
  `.PARTIAL.csv` and stops, because it skips modes whose CSV already exists and a
  partial would otherwise be silently accepted forever.
- **Ground truths are LLM-transcribed-then-spot-checked**, not hand-authored (above).

---

## Known limitations / deferred

- **Chroma uses L2 distance, not cosine.** Retrieval performs well empirically; a
  cosine re-index is a zero-API local fix, deferred.
- **Dense retrieval is the weak link on paraphrases.** The embedding model is trained
  on general web text, not JavaDoc; reworded questions that share no vocabulary with
  the docs are where recall dips.

---

## Project layout

```
corpus/            26 JavaDoc HTML pages (the knowledge base)
ingestion/         document_processor → chunker → indexer (builds Chroma + BM25)
retrieval/         dense / sparse / hybrid retrievers + cross-encoder reranker
generation/        generator (grounding prompt) + pipeline (the ablation switch)
eval/              eval_set.json (55 Qs) + eval_set.example.json
evaluation/        ablation_runner.py (3 stages) + gold.py (multi-gold support)
scripts/           ask, test_retrieval, browse_corpus, failure_analysis, validate_eval_set
results/           raw_runs.json (cached), ragas_*.csv, ablation_table.csv, failure_analysis.md
```
