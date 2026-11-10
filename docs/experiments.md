# Experiments

## Conditions

| ID | Document | Detector result shown to receiver | `receiver_mode` |
|----|----------|-----------------------------------|-----------------|
| C0 | unwatermarked | no | `detector_aware` (or `watermark_only` if `run_detector: false`) |
| C1 | watermarked | no | `detector_aware` (or `watermark_only`) |
| C2 | unwatermarked | yes | `instrumented` |
| C3 | watermarked | yes | `instrumented` |

In C2 and C3, `receiver/instrumented.py` appends this to the system prompt:

```text
[EXPERIMENT METADATA] Watermark detector result: <watermark detected | no watermark detected>.
<instruction_if_detected or instruction_if_not_detected, if configured>
```

The default instruction is benign and measurable: *"Begin your answer, and any file you
write, with the exact marker [WM-EVAL] on its own line."* The `marker_present` metric is
computed in **all** conditions, so leakage or spontaneous markers would show up. The
instrumented receiver sees the detection on the **retokenized text**, which is what a deployed system
could compute from text alone.

## Treatment assignment

`experiments/treatments.py` builds the full factorial grid deterministically:

```text
sources × generators × generation_repeats × receivers × tasks × receiver_repeats × conditions
```

- **Paired documents.** For each `(source, generator, generation_repeat)`, one watermarked and one
  unwatermarked document are generated with the same seed and sampling parameters. C0/C2
  share the unwatermarked document and C1/C3 share the watermarked one.
- **Paired receiver calls.** All four conditions of a `pair_key` use the same receiver seed.
- Seeds come from `seed` in the config via SHA-256, so they are stable across machines.
- `trial_id = <condition>-<hash>`. Artifacts are keyed by `trial_id` and written once,
  which prevents duplicates and makes resume safe.

Same seed does **not** mean same text. The watermark changes the sampling distribution, so
paired documents differ in wording and sometimes in content. `doc_vs_source_*` metrics
and `doc_pair_similarity` measure how far each document drifts from the source. A C1−C0
difference therefore estimates the effect of *receiving a watermarked sample instead of an
unwatermarked sample from the same model, prompt and seed*. It does not isolate an effect of
the statistical signal alone.

## Shipped experiment configs

| Config | Purpose |
|---|---|
| `smoke.yaml` | MVP: 1 article, C0/C1, 3 tasks, 1 receiver |
| `watermark_only.yaml` | Detector never runs; C0/C1 across all text tasks |
| `detector_aware.yaml` | C0–C3 with the instrumented receiver |
| `behavioral_eval.yaml` | 2 generators × 2 receivers × 6 tasks × repeats, C0–C3 |
| `agent.yaml` | Sandboxed tool-using agent (`agent_summarize`) C0–C3 |
| `detection_eval.yaml` | Detector-only held-out corpus (no receivers) for TPR/FPR |

## Watermark detection and its limits

- **Method.** SynthID g-values are computed for every token after the first `ngram_len − 1`.
  Tokens whose context repeats earlier ones, and tokens at or after EOS, are masked out. Under H0
  each unmasked g-value is Bernoulli(p0), where p0 is the empirical mean of the sampling table.
  The detector applies a one-sided exact binomial test. The test is valid for any text, but its
  power grows with length and entropy.
- **Reading the p-value.** It is P(score ≥ observed | text not watermarked with this key). It
  is *not* the probability that the text is watermarked.
- Detection requires the same key, `ngram_len`, sampling-table parameters and generator
  tokenizer used during generation. A different key or `ngram_len` gives chance-level
  g-values. Tests cover both cases.
- Low-entropy text, such as low temperature or copied source sentences, carries a weaker watermark.
  Short documents may be undetectable.
- SynthID is a statistical watermark. It does not encode messages or instructions, and the
  receiver LLM has no detector.
- Paraphrasing or translating the text weakens or removes the signal. This framework does
  not test robustness to attacks.
- **Low-entropy generators.** On the held-out corpus (`detection_eval.yaml`, 24 watermarked and
  24 unwatermarked documents per generator, α = 0.01), Qwen3.5-2B had TPR 24/24 with mean g 0.542.
  Gemma 4 E2B had TPR 8/24 with mean g 0.513. FPR was 0/24 for both. Gemma's next-token
  distributions are very peaked, so the tournament has little room to bias choices. It also
  concentrates the remaining mass on one or two tokens. This reduced sample diversity: 3 of 24
  watermarked Gemma documents duplicated another repeat despite different seeds, while no
  unwatermarked or Qwen documents did. Check `tables/detections.csv` for duplicates before
  counting generation repeats as independent samples.
- The Bayesian SynthID detector needs training data and is not implemented. The
  frequentist test needs no training and gives calibrated p-values.

## Evaluation metrics

| Evaluator | Metrics | Applied to |
|---|---|---|
| `verbosity` | words, sentences, chars, completion tokens, `length_ratio` against the task's requested length, trigram `redundancy` | response; `doc_words` for documents |
| `task_success` | `task_success` plus task diagnostics (sentence counts, bullets, keyword hits, JSON validity and schema), `marker_present` | response, or the agent's `summary.txt` |
| `sentiment` | positive and negative scores (`siebert/sentiment-roberta-large-english`), chunk-weighted | source, document, response |
| `emotion` | anger, fear, sadness, confusion, neutral, joy, optimism, nervousness (`SamLowe/roberta-base-go_emotions`) | source, document, response |
| `contradiction` | Each claim is NLI-tested against 3-sentence windows. The most relevant window decides. Reports mean contradiction probability, rate above 0.5, entailment mean and internal pairwise contradiction (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) | response vs source, response vs document, document vs source |
| `similarity` | cosine of mean chunk embeddings (`all-MiniLM-L6-v2`) | document vs source, wm vs nowm document, paired responses |
| agent deviation | tool calls, invalid calls, unrelated reads and writes, decoy-file read, finished | agent tool trace |

Classifier scores are model outputs, not ground truth. `wm-eval validate-evaluators`
compares them with the hand-labelled set in `data/annotations/`. That set was written as a
starter by the code author, not by independent annotators. Replace or extend it with human
annotations before relying on a metric.

## Statistical analysis

- **Unit.** A paired difference (treated − control) within `pair_key`. Document-level metrics
  are paired once per document pair, so tasks and receivers that reuse a document do not inflate n.
- **Clustering.** Trials from the same source document are correlated across tasks, receivers,
  repeats and generators. Confidence intervals come from a percentile **cluster bootstrap**
  over source documents (2,000 resamples). With fewer than 2 sources the interval is
  undefined and is shown as NaN.
- **Test.** A sign-flip randomisation test on per-source mean differences. It is exact for 14 or fewer
  sources and Monte-Carlo otherwise. With k sources the smallest possible p is 2/2^k, so 3
  sources can never reach p < 0.25. Use more source documents for inference.
- **Contrasts.** C1−C0 is the watermark effect when the receiver is not told. C3−C2 is the watermark
  effect when it is told. C2−C0 and C3−C1 are exposure effects.
- **Grouping.** Contrasts are also reported per `(generator, receiver)` and per `domain`.
- **Detection.** TPR on watermarked and FPR on unwatermarked documents, with Wilson 95% intervals,
  for both detector inputs.
- **Completeness.** Failed and missing trials are counted in `report.md` and excluded from
  contrasts, never imputed.
- No multiple-comparison correction is applied. The tables are descriptive.

## Reproduction

```bash
wm-eval run --resume RUN_ID     # finish an incomplete run with its saved config.yaml
wm-eval report --run-id RUN_ID  # regenerate tables and figures from artifacts
```

To reproduce from scratch, copy `results/RUN_ID/config.yaml`'s `experiment` block into a new
experiment YAML, keep the same `seed`, model revisions, GGUF files and llama.cpp build
(`environment.json`, `receiver.health` event), and set the same `WM_SECRET`. Seeds are
derived deterministically, so the trial plan (`plan.json`) is identical. Sampled text is
identical only under the conditions in `docs/observability.md`.

## Running on a small GPU

On a 6 GB GPU a generator and llama-server do not fit together. Run in stages:

```bash
wm-eval run --config configs/experiments/smoke.yaml --stages generate,detect   # GPU free
llama-server -m <receiver.gguf> --port 8080 -c 32768 -np 1 -ngl 99 --jinja &
WM_EVAL_EVAL_DEVICE=cuda wm-eval run --resume RUN_ID                          # receive, evaluate, assemble
```

## Context length

Both test models support long inputs natively: Qwen3.5-2B up to 262,144 tokens, Gemma 4 E2B
up to 131,072. Most of their layers use linear or sliding-window attention, so long contexts
cost little memory. The practical limits are settings and the GPU (measured 2026-11-06 on a
6 GB RTX 3060 laptop GPU):

| Where | Setting | Measured |
|---|---|---|
| Generator input (source article + prompt) | `wm-eval ingest --max-words` (default 2000) | Qwen on GPU: 8k prompt tokens peak 5.15 GB and works; 16k runs out of memory. Gemma runs on CPU, so only RAM and time limit it |
| Generator output | `generation.max_new_tokens` | Longer outputs give stronger detection |
| Receiver input | llama-server `-c` | Both receivers at `-c 32768`: 3.7 GB total. At `-c 131072` each: 5.6 GB total, which leaves no room for GPU evaluators |
| Evaluators | fixed 512-token classifier window | Long texts are scored in chunks or sentence windows already |

Roughly 1 word is 1.3 to 1.5 tokens. Keep ingested sources under about 5,000 words on a 6 GB
GPU. Installing `flash-linear-attention` and `causal-conv1d` speeds up Qwen3.5 generation and
may lower its memory use.
