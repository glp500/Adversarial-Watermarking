# Observability

## Events

`observability/events.py` provides one `Recorder` per run. It appends to
`results/RUN_ID/events.jsonl` and mirrors each line to the standard log.

```python
recorder.emit("watermark.detection.completed", doc_id=doc_id, data={"p_value_token_ids": p})
```

Each line has `ts` (UTC, microseconds), `event`, `run_id`, any identifiers passed as
keywords (`trial_id`, `doc_id`, `source_id`, `condition`, `receiver`, `generator`) and an
optional `data` object.

| Event | Emitted when |
|---|---|
| `experiment.started` / `experiment.planned` / `experiment.completed` / `experiment.failed` | Run lifecycle (failed includes the traceback) |
| `document.loaded` | Each source copied into the run |
| `generation.model_loading` / `generation.model_loaded` | Model spec and resolved metadata (class, device map, commit) |
| `generation.started` / `generation.completed` / `generation.failed` | Per document, with token count, finish reason, duration |
| `watermark.detection.started` / `.completed` / `.failed` | Per document, with p-values for both input kinds |
| `receiver.health` | `/health` and `/props` of each server (model path, build) |
| `instrumentation.decided` | The exact text the instrumented adapter injects (or `applied: false`) |
| `receiver.inference.started` / `.completed` / `.failed` / `receiver.skipped` | Per trial, with token counts and duration or error |
| `evaluation.started` / `evaluation.completed` / `evaluation.retrying` | Per evaluation artifact, with per-evaluator errors. Errored evaluations are retried on resume and the old attempt moves to `failures/eval__*` |
| `results.assembled`, `report.started`, `report.completed` | Aggregation steps |
| `doctor.completed`, `evaluators.validated` | `wm-eval doctor` / `validate-evaluators` results (written to `results/_commands/`) |

## Secrets

- The watermark keys are derived in memory from the `WM_SECRET` env var (`.env`). Only
  `key_id` and the non-secret SynthID parameters are written.
- The recorder redacts values under keys that look secret (`keys`, `secret`, `api_key`, ...).
- The generation config recorded in artifacts is captured *before* the watermark config
  (which holds keys) is attached.
- The end-to-end test scans every file in a run directory for the secret and derived keys.

## Artifacts

```text
results/RUN_ID/
├── config.yaml          fully resolved config (re-used verbatim on --resume)
├── environment.json     Python, packages, platform, CUDA, GPUs
├── plan.json            every planned trial with ids and seeds
├── events.jsonl         append-only execution trace
├── trials.parquet       one row per trial (also tables/trials.csv)
├── artifacts/
│   ├── sources/         copies of input documents
│   ├── generated/       prompt, prompt token ids, generated token ids, text, metadata
│   ├── detections/      detector output on token ids and on retokenized text
│   ├── instructions/    instrumented-adapter decision per trial
│   ├── responses/       full request and raw response payloads (+ tool traces)
│   ├── failures/        every failed attempt: gen__, det__, rec__, eval__ (timestamped, never overwritten)
│   └── evaluations/     source__, doc__, docpair__, trial__, pair__ metric files
├── metrics/             summary.json, comparisons.json
├── tables/              CSV + Parquet comparison and detection tables
├── figures/             PNG figures
└── report.md
```

Every trial row stores relative paths to its `source_artifact`, `generated_artifact` and
`response_artifact`. Every evaluation file records evaluator versions (model id @ commit)
and any evaluator errors.

## Inspecting a run without re-running models

```bash
jq -c 'select(.event|test("failed"))' results/RUN_ID/events.jsonl
python -c "import pandas as pd; print(pd.read_parquet('results/RUN_ID/trials.parquet').head())"
cat results/RUN_ID/artifacts/responses/<trial_id>.json
```

## Sources of nondeterminism

- **Transformers sampling on GPU.** `set_seed` makes CPU runs repeatable. CUDA kernels
  (and `device_map` offloading) can still differ across hardware and library versions.
- **llama.cpp.** A fixed `seed` is sent with every request. Results repeat only with the
  same build (`receiver.health.build_info`), GGUF file, context size and hardware, and with
  `-np 1` (parallel slots batch requests together).
- **Accelerate placement.** With `device: auto`, layers go to GPU or CPU depending on free
  memory at load time. The placement is recorded in `generation.model_loaded`.
- **Evaluator classifiers.** These are deterministic for a fixed model commit and device. Results can
  differ slightly between CPU and CUDA.
