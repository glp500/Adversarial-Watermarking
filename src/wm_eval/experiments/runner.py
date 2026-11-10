"""Experiment runner: generate -> detect -> receive -> evaluate -> assemble.

Each stage reads and writes write-once artifacts in the run directory, so any stage can be
re-run (resume) and only missing work is done. Failed attempts are stored under
artifacts/failures/ and retried on resume; successful results are never redone.
"""

from __future__ import annotations

import json
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from wm_eval.config import ResolvedConfig, load_resolved
from wm_eval.data.ingest import load_jsonl
from wm_eval.data.preprocess import normalize, truncate
from wm_eval.evaluation import suite
from wm_eval.experiments.tasks import TASKS, task_prompt
from wm_eval.experiments.treatments import PlannedDoc, assign_trials, plan_documents
from wm_eval.observability.events import Recorder
from wm_eval.observability.storage import (
    RunDir,
    environment_info,
    new_run_id,
    write_once,
    write_yaml_once,
)
from wm_eval.schemas import (
    DetectionResult,
    GenerationRequest,
    GenerationResult,
    SourceDocument,
    Trial,
)

STAGES = ("generate", "detect", "receive", "evaluate", "assemble")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _err(e: BaseException) -> dict[str, str]:
    return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


# ----------------------------------------------------------------------------- setup


def create_run(cfg: ResolvedConfig, root: str | Path | None = None) -> RunDir:
    rd = RunDir(root or cfg.settings.output_root, new_run_id(cfg.experiment.name)).create()
    write_yaml_once(rd.path / "config.yaml", cfg.model_dump(mode="json"))
    write_once(rd.path / "environment.json", environment_info())
    return rd


def open_run(run_id: str, root: str | Path) -> tuple[RunDir, ResolvedConfig]:
    rd = RunDir(root, run_id)
    if not rd.path.exists():
        raise FileNotFoundError(f"no run directory {rd.path}")
    return rd, load_resolved(rd.path / "config.yaml")


def load_sources(cfg: ResolvedConfig, rd: RunDir, rec: Recorder) -> dict[str, SourceDocument]:
    docs = load_jsonl(cfg.experiment.sources, cfg.experiment.source_ids)
    for d in docs:
        if not rd.has("sources", d.source_id):
            rd.save("sources", d.source_id, d)
        rec.emit(
            "document.loaded",
            source_id=d.source_id,
            data={"chars": len(d.text), "domain": d.domain},
        )
    # The run's own copy is authoritative (protects reproducibility if the input file changes).
    return {d.source_id: SourceDocument(**rd.load("sources", d.source_id)) for d in docs}


def plan_trials(cfg: ResolvedConfig, rd: RunDir, sources: dict[str, SourceDocument]) -> list[Trial]:
    trials = assign_trials(cfg.experiment, list(sources))
    plan = rd.path / "plan.json"
    data = [t.model_dump() for t in trials]
    if plan.exists():
        if json.loads(plan.read_text()) != data:
            raise RuntimeError("trial plan differs from the saved plan.json; refusing to resume")
    else:
        with open(plan, "x") as f:
            json.dump(data, f, indent=1)
    return trials


# ----------------------------------------------------------------------------- stages


def stage_generate(
    cfg: ResolvedConfig, rd: RunDir, rec: Recorder, sources, docs: list[PlannedDoc]
) -> None:
    from wm_eval.generation.generator import generate
    from wm_eval.generation.models import free_memory, load_model, model_metadata

    exp = cfg.experiment
    by_gen: dict[str, list[PlannedDoc]] = defaultdict(list)
    for t in docs:
        if not rd.has("generated", t.doc_id):
            by_gen[t.generator].append(t)
    for gen, todo in by_gen.items():
        spec = cfg.generators[gen]
        rec.emit("generation.model_loading", generator=gen, data=spec.model_dump())
        try:
            model, tok = load_model(spec)
        except Exception as e:
            rec.emit("generation.failed", generator=gen, data={"stage": "load_model", **_err(e)})
            continue
        meta = model_metadata(spec, model)
        rec.emit("generation.model_loaded", generator=gen, data=meta)
        for t in todo:
            src = sources[t.source_id]
            article = truncate(normalize(src.text), exp.max_document_chars)
            req = GenerationRequest(
                source_id=t.source_id, prompt=exp.rewrite_prompt.replace("{article}", article),
                generator=gen, watermark_enabled=t.watermark_enabled, seed=t.generation_seed,
            )  # fmt: skip
            ids = {"doc_id": t.doc_id, "source_id": t.source_id, "generator": gen}
            rec.emit("generation.started", **ids, data={"watermark_enabled": t.watermark_enabled,
                                                       "seed": t.generation_seed})  # fmt: skip
            try:
                res = generate(model, tok, req, t.doc_id, rd.run_id, spec, exp.generation,
                               cfg.watermark, meta)  # fmt: skip
            except Exception as e:
                rd.save("failures", f"gen__{t.doc_id}__{_stamp()}", {**ids, **_err(e)})
                rec.emit("generation.failed", **ids, data=_err(e))
                continue
            rd.save("generated", t.doc_id, res)
            rec.emit("generation.completed", **ids, data={
                "n_tokens": len(res.token_ids), "finish_reason": res.finish_reason,
                "duration_s": round(res.duration_s, 3)})  # fmt: skip
        del model, tok  # drop the only references so the next generator gets the GPU
        free_memory()


def stage_detect(cfg: ResolvedConfig, rd: RunDir, rec: Recorder) -> None:
    from wm_eval.generation.models import load_tokenizer
    from wm_eval.watermark.detector import detect, detect_text

    tokenizers: dict[str, Any] = {}
    for path in rd.list("generated"):
        g = GenerationResult(**json.loads(path.read_text()))
        if rd.has("detections", g.doc_id):
            continue
        spec = cfg.generators[g.generator]
        tok_id = spec.tokenizer_id or spec.hf_id
        ids = {"doc_id": g.doc_id, "source_id": g.source_id, "generator": g.generator}
        rec.emit(
            "watermark.detection.started",
            **ids,
            data={"watermark_config_id": cfg.watermark.config_id},
        )
        try:
            if g.generator not in tokenizers:
                tokenizers[g.generator] = load_tokenizer(spec)
            on_ids = detect(
                g.doc_id, g.token_ids, cfg.watermark, tok_id, g.metadata.get("eos_token_ids", [])
            )
            on_text = detect_text(g.doc_id, g.text, tokenizers[g.generator], cfg.watermark, tok_id)
        except Exception as e:
            rd.save("failures", f"det__{g.doc_id}__{_stamp()}", {**ids, **_err(e)})
            rec.emit("watermark.detection.failed", **ids, data=_err(e))
            continue
        rd.save("detections", g.doc_id, {
            "doc_id": g.doc_id, "generated_artifact": str(path.relative_to(rd.path)),
            "watermark_enabled": g.watermark_enabled,  # ground-truth label, for TPR/FPR
            "token_ids": on_ids.model_dump(), "retokenized_text": on_text.model_dump()})  # fmt: skip
        rec.emit("watermark.detection.completed", **ids, data={
            "p_value_token_ids": on_ids.p_value, "detected_token_ids": on_ids.watermark_detected,
            "p_value_retokenized": on_text.p_value, "detected_retokenized": on_text.watermark_detected,
            "mean_g": on_ids.mean_g, "n_scored": on_ids.n_scored})  # fmt: skip


def stage_receive(
    cfg: ResolvedConfig, rd: RunDir, rec: Recorder, sources, trials: list[Trial]
) -> None:
    from wm_eval.agent.simulated import run_agent
    from wm_eval.receiver.inference import build_messages, run_receiver
    from wm_eval.receiver.instrumented import instruction_for
    from wm_eval.receiver.llama_client import LlamaClient

    exp = cfg.experiment
    clients = {name: LlamaClient(spec) for name, spec in cfg.receivers.items()}
    for name, client in clients.items():
        rec.emit("receiver.health", receiver=name, data=client.health())
    for t in trials:
        if rd.has("responses", t.trial_id):
            continue
        ids = {"trial_id": t.trial_id, "source_id": t.source_id, "condition": t.condition,
               "receiver": t.receiver, "doc_id": t.doc_id}  # fmt: skip
        if not rd.has("generated", t.doc_id):
            rec.emit("receiver.skipped", **ids, data={"reason": "generated document missing"})
            continue
        doc = rd.load("generated", t.doc_id)
        detection = None
        if exp.run_detector and rd.has("detections", t.doc_id):
            # The instrumented receiver sees what a deployed system could compute from text
            # alone: detection on the retokenized document.
            detection = DetectionResult(**rd.load("detections", t.doc_id)["retokenized_text"])
        if t.expose_detection and detection is None:
            rec.emit(
                "receiver.skipped",
                **ids,
                data={"reason": "detection missing for exposed condition"},
            )
            continue
        instr = instruction_for(t.expose_detection, detection, exp.instrumentation)
        if not rd.has("instructions", t.trial_id):
            rd.save("instructions", t.trial_id, {
                "trial_id": t.trial_id, "condition": t.condition, "receiver_mode": t.receiver_mode,
                "detection_doc_id": t.doc_id if t.expose_detection else None,
                "detection_p_value": detection.p_value if (t.expose_detection and detection) else None,
                **instr.model_dump()})  # fmt: skip
            rec.emit("instrumentation.decided", **ids, data=instr.model_dump())
        src = sources[t.source_id]
        prompt = task_prompt(t.task, src)
        rec.emit(
            "receiver.inference.started", **ids, data={"task": t.task, "seed": t.receiver_seed}
        )
        if t.task == "agent_summarize":
            res = run_agent(clients[t.receiver], t.trial_id, rd.run_id, t.receiver, prompt, doc["text"],
                            instr, exp.receiver_params, t.receiver_seed, exp.agent_max_steps)  # fmt: skip
        else:
            msgs = build_messages(prompt, doc["text"], instr)
            res = run_receiver(clients[t.receiver], t.trial_id, rd.run_id, t.receiver, msgs,
                               exp.receiver_params, t.receiver_seed)  # fmt: skip
        if res.status == "ok":
            rd.save("responses", t.trial_id, res)
            rec.emit("receiver.inference.completed", **ids, data={
                "completion_tokens": res.completion_tokens, "prompt_tokens": res.prompt_tokens,
                "duration_s": round(res.duration_s, 3),
                "tool_calls": len(res.tool_trace) if res.tool_trace is not None else None})  # fmt: skip
        else:
            rd.save("failures", f"rec__{t.trial_id}__{_stamp()}", res)
            rec.emit("receiver.inference.failed", **ids, data={"error": res.error})


def _save_eval(rd: RunDir, rec: Recorder, name: str, fn, **ids) -> None:
    """Evaluate once. An evaluation that recorded evaluator errors (e.g. CUDA OOM) is retried
    on resume: the errored attempt is moved to artifacts/failures/ (kept, not overwritten)."""
    if rd.has("evaluations", name):
        prev = rd.load("evaluations", name)
        if not prev.get("errors"):
            return
        rd.artifact("evaluations", name).rename(
            rd.artifact("failures", f"eval__{name}__{_stamp()}")
        )
        rec.emit(
            "evaluation.retrying", evaluation=name, **ids, data={"previous_errors": prev["errors"]}
        )
    rec.emit("evaluation.started", evaluation=name, **ids)
    res = fn()
    rd.save("evaluations", name, res)
    rec.emit("evaluation.completed", evaluation=name, **ids,
             data={"duration_s": round(res.duration_s, 3), "errors": res.errors})  # fmt: skip


def stage_evaluate(
    cfg: ResolvedConfig, rd: RunDir, rec: Recorder, sources, trials: list[Trial]
) -> None:
    exp = cfg.experiment
    ev = exp.evaluators
    marker = exp.instrumentation.marker if exp.instrumentation else None
    for sid, src in sources.items():
        _save_eval(
            rd, rec, f"source__{sid}", lambda src=src: suite.evaluate_source(ev, src), source_id=sid
        )
    docs: dict[str, dict] = {}
    for path in rd.list("generated"):
        d = json.loads(path.read_text())
        docs[d["doc_id"]] = d
        _save_eval(rd, rec, f"doc__{d['doc_id']}",
                   lambda d=d: suite.evaluate_document(ev, d["doc_id"], d["text"], sources[d["source_id"]]),
                   doc_id=d["doc_id"])  # fmt: skip
    # watermarked vs unwatermarked document similarity (source fidelity of the pair)
    for d in docs.values():
        if d["watermark_enabled"]:
            other = d["doc_id"].removesuffix("__wm") + "__nowm"
            if other in docs:
                _save_eval(rd, rec, f"docpair__{d['doc_id']}",
                           lambda d=d, o=docs[other]: suite.evaluate_pair(ev, d["doc_id"], d["text"], o["text"]),
                           doc_id=d["doc_id"])  # fmt: skip
    responses: dict[str, dict] = {}
    for t in trials:
        if not rd.has("responses", t.trial_id):
            continue
        r = responses[t.trial_id] = rd.load("responses", t.trial_id)
        _save_eval(rd, rec, f"trial__{t.trial_id}", lambda t=t, r=r: suite.evaluate_response(
            ev, t.trial_id, t.task, r["text"], docs[t.doc_id]["text"], sources[t.source_id],
            TASKS[t.task].target_words, r.get("completion_tokens"), marker, r.get("workspace_files"),
            r.get("tool_trace")),
            trial_id=t.trial_id, condition=t.condition)  # fmt: skip
    # response consistency within each pair: C1 vs C0 and C3 vs C2
    by_pair = defaultdict(dict)
    for t in trials:
        by_pair[t.pair_key][t.condition] = t
    for conds in by_pair.values():
        for treated, control in (("C1", "C0"), ("C3", "C2")):
            a, b = conds.get(treated), conds.get(control)
            if a and b and a.trial_id in responses and b.trial_id in responses:
                _save_eval(rd, rec, f"pair__{a.trial_id}__{b.trial_id}",
                           lambda a=a, b=b: suite.evaluate_pair(
                               ev, f"{a.trial_id}|{b.trial_id}",
                               responses[a.trial_id]["text"], responses[b.trial_id]["text"]),
                           trial_id=a.trial_id)  # fmt: skip


# ----------------------------------------------------------------------------- assemble


def _metrics(rd: RunDir, name: str, prefix: str = "") -> dict[str, Any]:
    if not rd.has("evaluations", name):
        return {}
    e = rd.load("evaluations", name)
    return {f"metrics.{prefix}{k}": v for k, v in e["metrics"].items()}


def assemble(cfg: ResolvedConfig, rd: RunDir, rec: Recorder, trials: list[Trial]) -> pd.DataFrame:
    failures = defaultdict(list)
    for p in rd.list("failures"):
        if p.name.startswith("rec__"):
            failures[p.name.split("__")[1]].append(p.name)
    pair_sim: dict[str, float | None] = {}
    for p in rd.list("evaluations"):
        if p.name.startswith("pair__"):
            _, a, b = p.stem.split("__")
            v = json.loads(p.read_text())["metrics"].get("pair_similarity")
            pair_sim[a] = pair_sim[b] = v
    rows = []
    for t in trials:
        row: dict[str, Any] = t.model_dump()
        row["run_id"] = rd.run_id
        row["watermark_config_id"] = cfg.watermark.config_id
        row["n_failed_attempts"] = len(failures[t.trial_id])
        row["source_artifact"] = f"artifacts/sources/{t.source_id}.json"
        row["domain"] = rd.load("sources", t.source_id)["domain"]
        row["generated_artifact"] = (
            f"artifacts/generated/{t.doc_id}.json" if rd.has("generated", t.doc_id) else None
        )
        row["response_artifact"] = (
            f"artifacts/responses/{t.trial_id}.json" if rd.has("responses", t.trial_id) else None
        )
        row["status"] = (
            "ok" if row["response_artifact"] else "failed" if failures[t.trial_id]
            else "missing_document" if not row["generated_artifact"] else "pending"
        )  # fmt: skip
        if rd.has("generated", t.doc_id):
            g = rd.load("generated", t.doc_id)
            row["generator_hf_id"] = g["metadata"]["model"]["hf_id"]
            row["generator_revision"] = g["metadata"]["model"]["revision"]
            row["doc_tokens"] = len(g["token_ids"])
            row["doc_finish_reason"] = g["finish_reason"]
        if rd.has("detections", t.doc_id):
            det = rd.load("detections", t.doc_id)
            for kind, short in (("token_ids", "ids"), ("retokenized_text", "text")):
                row[f"watermark_p_value_{short}"] = det[kind]["p_value"]
                row[f"watermark_detected_{short}"] = det[kind]["watermark_detected"]
            row["watermark_mean_g"] = det["token_ids"]["mean_g"]
            row["watermark_p_value"] = row["watermark_p_value_ids"]
            row["watermark_detected"] = row["watermark_detected_ids"]
        if row["response_artifact"]:
            r = rd.load("responses", t.trial_id)
            row["receiver_completion_tokens"] = r.get("completion_tokens")
            row["receiver_duration_s"] = r.get("duration_s")
        if rd.has("instructions", t.trial_id):
            row["instruction_applied"] = rd.load("instructions", t.trial_id)["applied"]
        spec = cfg.receivers[t.receiver]
        row["receiver_model"] = spec.gguf or spec.model
        row.update(_metrics(rd, f"trial__{t.trial_id}"))
        row.update(_metrics(rd, f"doc__{t.doc_id}"))
        row.update(_metrics(rd, f"source__{t.source_id}"))
        if t.watermark_enabled:
            row.update(_metrics(rd, f"docpair__{t.doc_id}", "doc_"))
        else:
            row.update(_metrics(rd, f"docpair__{t.doc_id.removesuffix('__nowm')}__wm", "doc_"))
        row["metrics.paired_response_similarity"] = pair_sim.get(t.trial_id)
        rows.append(row)
    base_cols = [*Trial.model_fields, "run_id", "status", "n_failed_attempts", "domain"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=base_cols)
    df.to_parquet(rd.path / "trials.parquet", index=False)
    df.to_csv(rd.path / "tables" / "trials.csv", index=False)
    det_rows = []
    for p in rd.list("detections"):
        d = json.loads(p.read_text())
        for kind in ("token_ids", "retokenized_text"):
            gen = rd.load("generated", d["doc_id"])["generator"]
            det_rows.append({"doc_id": d["doc_id"], "generator": gen, "watermark_enabled": d["watermark_enabled"],
                             **d[kind]})  # fmt: skip
    det_df = pd.DataFrame(det_rows)
    if len(det_df):
        det_df.to_parquet(rd.path / "tables" / "detections.parquet", index=False)
        det_df.to_csv(rd.path / "tables" / "detections.csv", index=False)
    rec.emit(
        "results.assembled",
        data={"n_trials": len(df), "status_counts": df["status"].value_counts().to_dict()},
    )
    return df


# ----------------------------------------------------------------------------- entry point


def run(
    cfg: ResolvedConfig | None = None,
    run_id: str | None = None,
    root: str | Path | None = None,
    stages: tuple[str, ...] = STAGES,
) -> RunDir:
    """New run (cfg given) or resume (run_id given; uses the run's saved config)."""
    if run_id:
        rd, cfg = open_run(run_id, root or (cfg.settings.output_root if cfg else "results"))
    else:
        assert cfg is not None
        rd = create_run(cfg, root)
    rec = Recorder(rd.events, rd.run_id)
    rec.emit("experiment.started", data={"experiment": cfg.experiment.name, "resumed": bool(run_id),
                                        "stages": list(stages)})  # fmt: skip
    try:
        sources = load_sources(cfg, rd, rec)
        trials = plan_trials(cfg, rd, sources)
        rec.emit(
            "experiment.planned",
            data={"n_trials": len(trials), "conditions": cfg.experiment.conditions},
        )
        if "generate" in stages:
            stage_generate(cfg, rd, rec, sources, plan_documents(cfg.experiment, list(sources)))
        if "detect" in stages and cfg.experiment.run_detector:
            stage_detect(cfg, rd, rec)
        if "receive" in stages:
            stage_receive(cfg, rd, rec, sources, trials)
        if "evaluate" in stages:
            stage_evaluate(cfg, rd, rec, sources, trials)
        if "assemble" in stages:
            df = assemble(cfg, rd, rec, trials)
            from wm_eval.analysis.statistics import write_summary

            write_summary(rd, df)
    except Exception as e:
        rec.emit("experiment.failed", data=_err(e))
        raise
    rec.emit("experiment.completed")
    return rd
