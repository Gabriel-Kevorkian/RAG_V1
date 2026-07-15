# Failure analysis

_Buckets are computed from passage-level retrieval facts and refusal detection, not from RAGAS judge scores. See the header of `scripts/failure_analysis.py` for why the previous version's buckets were inverted._

## 0. Abstention: does it refuse when the corpus cannot answer?

- `dense_only` — refused **10 / 10**
- `sparse_only` — refused **10 / 10**
- `hybrid` — refused **10 / 10**
- `hybrid_rerank` — refused **10 / 10**

The eval set has **55** questions. RAGAS scores **45** of them: it needs a `reference` to compare against, and the 10 `unanswerable` questions have none, so it drops them silently. Every previous version of *this report* inherited that blind spot and analysed 45 questions while believing it had analysed the set.

To be precise about who missed what: `ablation_runner.py` never missed it. It computes `refusal_rate_unanswerable` and has been printing **1.0** in `ablation_table.csv` the whole time. It was the *failure analysis* — the file whose entire job is to ask what went wrong — that quietly scoped itself to the judged subset and never looked at the other ten.

Those 10 ask what the system does when the answer is simply not in the corpus — the difference between a search box and a liar. **All four configurations refused all ten.** Zero hallucinations, including from `sparse_only`, which is the weakest retriever here and had the most opportunity to bluff.

## 1. Retrieval, measured honestly (file-level vs passage-level)

```
               file_hit  passage_hit
mode                                
dense_only        0.978        0.911
sparse_only       0.956        0.844
hybrid            0.978        0.911
hybrid_rerank     1.000        0.956
```

`file_hit` = an acceptable JavaDoc **file** appeared among the retrieved chunks.
`passage_hit` = an acceptable gold **sentence** (`source_passage` / `alt_golds`, which have been in `eval_set.json` all along) appeared in the retrieved text.

The gap between these two columns is the whole failure analysis. The full system scores a perfect **1.000 on file_hit** — which is what the old report proudly printed — and **0.956 on passage_hit**. `java.util.TreeSet.html` is a very large page; pulling *a* chunk of it is not the same as pulling the *one sentence* that answers the question. File-level hit rate is a metric that cannot fail, and a metric that cannot fail is not measuring anything.

Right file, wrong sentence, in `hybrid_rerank`: **`p06`, `p07`** — 2 of 45. Every one of these was invisible to the old file-level metric.

## 2. Diagnosis: whose fault is each failure?

- **1** — grounding failure? (terse answer -- suspect judge artifact)
- **1** — answered without the gold passage -- ADJUDICATE
- **1** — retrieval miss (model correctly refused)
- **42** — ok

These take opposite repairs. Retrieval misses want better chunking and a bigger candidate pool. Grounding failures want a stricter prompt or a stronger model. Bad gold labels want an eval-set fix and no code change at all. Tuning the retriever to fix a prompt bug is a week you do not get back — and the previous version of this file would have sent me to do exactly that.

`ADJUDICATE` is not a hedge, it is the honest output of a judge-free test. When the model answers without the gold passage, the string comparison cannot tell me whether the gold label was too narrow or the model was wrong — that needs a human to read the chunks. §3 is that reading. It ran on three such questions: two (`p01`, `p10`) were bad gold labels and have been fixed in the eval set, which is why they no longer appear above. The one that remains, `p06`, is a genuine model error.


**`p06`** (paraphrase_heavy) — _answered without the gold passage -- ADJUDICATE_
- Q: If I turn an array into a list, can I add elements to it afterwards?
- gold: `java.util.Arrays.html` → "Returns a fixed-size list backed by the specified array"
- retrieved: ['java.util.ArrayList.html', 'java.util.Arrays.html', 'java.util.LinkedList.html'] | passage_hit **False**
- A: Yes, the list returned by `Arrays.asList` is modifiable [java.util.Arrays.html].
- scores: faith 0.50, relevancy 0.76, precision 0.20, recall 0.00

**`s14`** (simple) — _grounding failure? (terse answer -- suspect judge artifact)_
- Q: What does ArrayDeque.poll() return when the deque is empty?
- gold: `java.util.ArrayDeque.html` → "or returns null if this deque is empty"
- retrieved: ['java.util.ArrayDeque.html', 'java.util.LinkedList.html'] | passage_hit **True**
- A: It returns `null` [java.util.ArrayDeque.html].
- scores: faith 0.00, relevancy 0.86, precision 1.00, recall 1.00

**`p07`** (paraphrase_heavy) — _retrieval miss (model correctly refused)_
- Q: What happens when I put an object that cannot be ordered into a sorted set?
- gold: `java.util.TreeSet.html` → "an element to the set that violates this constraint, the add call will throw a ClassCastException"
- retrieved: ['java.util.Arrays.html', 'java.util.Collections.html', 'java.util.Comparator.html', 'java.util.PriorityQueue.html', 'java.util.TreeSet.html'] | passage_hit **False**
- A: insufficient context
- scores: faith 0.00, relevancy 0.00, precision 0.64, recall 1.00

## 3. Read the chunks. Every time.

Five questions come out of §2 as failures or adjudications. I opened the retrieved chunks for every one. **Not a single one was the failure the metrics said it was.**

| id | RAGAS says | actually |
|---|---|---|
| `p07` | worst question in the set (faith 0.00, relevancy 0.00) | **the system behaved perfectly.** `TreeSet.html` was retrieved, but the chunk was the `synchronizedSortedSet` boilerplate, not the constructor prose that says `add` throws `ClassCastException`. The gold sentence exists in the corpus and was never pulled. The model said `insufficient context` — the correct response to missing evidence — and RAGAS scored a refusal as a hallucination. A **retrieval** bug, punished as a generation bug. |
| `s14` | faithfulness **0.00** — total hallucination | the answer is `It returns null`, and chunk 0 reads *"or returns null if this deque is empty"*. It is verbatim correct. RAGAS decomposes an answer into claims and verifies each; a three-word answer yields no claim to verify and the score collapses to zero. A **judge artifact**. Note it flipped 1.00 → 0.00 purely from reranking (§4) — the same right answer, scored differently on two runs. |
| `p10` | recall **0.00** — retriever failed | the model answered with `String.join`, cited `java.lang.String.html`, and was faithful (1.00) and right. The eval set had written down `Collectors.joining` as the only acceptable source. Two valid answers, one gold label. An **eval-set** bug — no code change would fix it, and 'fixing' the retriever to chase it would make the system worse. **FIXED:** gold widened to accept both; recall 0.00 → **1.00**, precision 0.00 → **1.00**. |
| `p06` | retriever failure (recall 0.00) | **the only genuine model error in the set,** and it is a good one. The gold sentence (*"Returns a fixed-size list backed by the specified array"*) was NOT retrieved — but the chunk that *was* retrieved contains both *"a convenient way to create a **fixed-size** list"* and *"The list returned by this method **is modifiable**"*. The model read the second and answered "Yes, you can add elements." You cannot: `add()` throws `UnsupportedOperationException`. The JavaDoc's own wording is the trap — *modifiable* means `set()` works, not `add()` — and the model walked straight into it. Half retrieval near-miss, half genuine misreading. |
| `p01` | faith 1.00, recall 1.00 — looks **fine**, never flagged before | asked *"how do I read a file one line at a time?"*, the model answered `Files.lines()`. Correct, idiomatic, well-cited. The gold label says `BufferedReader`. Same story as `p10`: a **second** eval-set bug, and the old file-level metric could not see it because `BufferedReader.html` *was* retrieved — just not the sentence. Passage-level hit is what surfaced it. **FIXED:** gold widened to accept both; under the full system its precision went 0.42 → **1.00**. |

> The scoreboard said: 2 retriever failures, 2 LLM failures. The chunks said: **1 retrieval miss, 1 genuine model error, 2 broken gold labels, 1 judge artifact** — and a correct refusal filed as the worst answer in the run. Every single verdict the metrics handed down named the wrong culprit, and the passage-level check found a sixth problem (`p01`) that scored 1.00/1.00 and looked perfect.
>
> This is not an argument against RAGAS. It is an argument against reading RAGAS without opening the chunks underneath it. **The single genuine model bug in 55 questions is `p06`, and no metric in the suite pointed at it.**

**What was actually changed.** Only the eval set. `p01` and `p10` now carry `alt_golds` (see `evaluation/gold.py`), the retrieval metrics accept any gold, and the two questions were re-judged on the only two metrics a gold label can move — `context_precision` and `context_recall`. `faithfulness` and `answer_relevancy` never see the reference and were left untouched on purpose: re-rolling 43 unchanged questions through a judge that scores the same answer 1.00 one run and 0.00 the next would have buried the fix under fresh noise.

**The retriever was never touched, and the numbers went up anyway** — because two of the four failures were never in the retriever. That is the whole argument for reading the chunks before you start optimising:

```
                     context_precision      context_recall
dense_only            0.688  ->  0.694     0.904  ->  0.926
sparse_only           0.681  ->  0.700     0.815  ->  0.859
hybrid                0.743  ->  0.761     0.889  ->  0.911
hybrid_rerank         0.746  ->  0.781     0.944  ->  0.967
```
The ranking is unchanged and `hybrid_rerank` still wins every column, so the Day 8 conclusion holds. But ~3.5 points of the full system's context_precision was never a system deficiency at all — it was a bad ruler.

## 4. Worst 8 questions in the full system

```
 id       query_type  mean_score  faithfulness  answer_relevancy  context_precision  context_recall file_hit passage_hit
p06 paraphrase_heavy       0.364           0.5             0.758              0.200             0.0     True       False
p07 paraphrase_heavy       0.410           0.0             0.000              0.639             1.0     True       False
p05 paraphrase_heavy       0.712           1.0             0.848              0.500             0.5     True        True
s14           simple       0.715           0.0             0.858              1.000             1.0     True        True
s15           simple       0.795           1.0             0.980              0.200             1.0     True        True
p03 paraphrase_heavy       0.803           1.0             0.887              0.325             1.0     True        True
k10    keyword_heavy       0.818           1.0             0.938              0.333             1.0     True        True
s21           simple       0.825           1.0             0.966              0.333             1.0     True        True
```

## 5. What reranking changed, question by question

Reranking gained the gold **passage** on **2** questions and lost it on **0**.

All four modes cut to the same top 5, but the un-reranked ones take the FIRST 5 of the 20 candidates while the reranker RE-PICKS which 5 survive. So it does not merely reorder the window — it changes what is in it. That is why its gains land on recall-flavoured metrics, not on context_precision as I predicted on Day 4.

Read the `faithfulness` column with suspicion, though: `s14` and `p07` swing a full **-1.000** here, and §3 shows both swings are judge noise on a correct answer and a correct refusal. Some of the cross-encoder's apparent effect on the headline table is the judge changing its mind, not the system changing its behaviour.

```
     faithfulness  answer_relevancy  context_precision  context_recall  passage_gained  passage_lost
id                                                                                                  
s01         0.000             0.000              0.300             0.0           False         False
s03         0.000             0.000             -0.083             0.0           False         False
s04         0.000             0.000             -0.633             0.0           False         False
s06         0.000            -0.028              0.000             0.0           False         False
s07         0.000            -0.040             -0.250             0.0           False         False
s09         1.000             0.979              0.750             1.0            True         False
s10         0.000             0.000             -0.300             0.0           False         False
s11         0.000             0.000              0.306             0.0           False         False
s13         0.000             0.032              0.000             0.0           False         False
s14        -1.000             0.000              0.675             0.0           False         False
s15         1.000             0.980              0.200             1.0            True         False
s16         0.000            -0.026             -0.217             0.0           False         False
s17         0.000             0.000             -0.050             0.0           False         False
s18         0.000             0.027             -0.133             0.0           False         False
s19         0.000             0.000              0.361             0.0           False         False
s20         0.000             0.021              0.000             0.5           False         False
s21         0.667             0.000             -0.250             0.0           False         False
s24         0.000             0.000              0.411             0.0           False         False
k01         0.333             0.207              0.050             0.0           False         False
k02         0.000             0.126              0.083             0.0           False         False
k03         0.000             0.000              0.304             0.0           False         False
k04         0.000             0.000              0.250             0.0           False         False
k05         0.000             0.000             -0.054             0.0           False         False
k06         0.000             0.002             -0.583             0.0           False         False
k07         0.000            -0.016             -0.217             0.0           False         False
k08         0.000             0.017             -0.083             0.0           False         False
k10         0.000            -0.030             -0.167             0.0           False         False
p01         0.000             0.000              0.500             0.0           False         False
p02         0.000             0.017             -0.300             0.0           False         False
p03         0.000             0.017             -0.092             0.0           False         False
p04         0.333             0.015              0.333             0.0           False         False
p05         0.000             0.022              0.000             0.0           False         False
p06         0.500             0.758              0.200             0.0           False         False
p07        -1.000            -0.809             -0.278             0.0           False         False
p08         0.000             0.000             -0.133             0.0           False         False
p09         0.000             0.042             -0.167             0.0           False         False
p10         0.000             0.026              0.167             0.0           False         False
```

## 6. Every metric, every config

```
               faithfulness  answer_relevancy  context_precision  context_recall  file_hit  passage_hit  abstention
mode                                                                                                               
dense_only            0.888             0.822              0.694           0.926     0.978        0.911         1.0
sparse_only           0.806             0.769              0.700           0.859     0.956        0.844         1.0
hybrid                0.893             0.837              0.761           0.911     0.978        0.911         1.0
hybrid_rerank         0.933             0.889              0.781           0.967     1.000        0.956         1.0
```

`hybrid_rerank` is still the best configuration on every column, so the headline conclusion survives. What does not survive is the reason I believed it: `file_hit` 1.000 was never the win it looked like, and roughly half the faithfulness gap between configs is judge noise on four questions.