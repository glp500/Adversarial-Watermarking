# Evaluator validation set

`evaluator_gold.jsonl` holds small hand-labelled examples for checking the classifier evaluators:

| kind | fields | n |
|---|---|---|
| `sentiment` | `text`, `label` (positive/negative) | 12 |
| `contradiction` | `premise`, `hypothesis`, `label` (true = contradiction) | 12 |
| `emotion` | `text`, `label` (one of anger, fear, sadness, confusion, neutral) | 10 |

Provenance: written by the code author (an AI assistant) as a starter set and not
independently annotated. It is enough to catch a broken or mis-wired evaluator, but not to
estimate accuracy on this study's texts. Add examples drawn from real receiver responses,
labelled by at least two people, before relying on any classifier metric.

Run `wm-eval validate-evaluators` to score the evaluators against this file.
