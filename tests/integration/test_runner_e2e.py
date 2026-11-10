"""Whole pipeline with a tiny in-memory generator and a fake llama.cpp server."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.conftest import TEST_SECRET
from wm_eval.config import derive_keys
from wm_eval.experiments import runner
from wm_eval.generation import models as gen_models
from wm_eval.observability.events import read_events
from wm_eval.receiver.llama_client import LlamaClient


@pytest.fixture
def fake_world(monkeypatch, tiny_lm):
    model, tok = tiny_lm
    monkeypatch.setattr(gen_models, "load_model", lambda spec: (model, tok))
    monkeypatch.setattr(gen_models, "load_tokenizer", lambda spec: tok)
    sent = []
    state = {"fail_next": 0}

    def send(self, request):
        sent.append(request)
        if state["fail_next"] > 0:
            state["fail_next"] -= 1
            raise ConnectionError("server down")
        content = '{"title": "t", "main_finding": "m", "sample_size": 48}' if "JSON" in request["messages"][1]["content"] \
            else "First sentence here. Second sentence here."  # fmt: skip
        if "WM-EVAL" in request["messages"][0]["content"]:
            content = "[WM-EVAL]\n" + content
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 12}}, 0.01  # fmt: skip

    monkeypatch.setattr(LlamaClient, "send", send)
    monkeypatch.setattr(
        LlamaClient, "health", lambda self, timeout=3.0: {"ok": True, "model_path": "fake.gguf"}
    )
    return SimpleNamespace(sent=sent, state=state)


def test_full_run_resume_and_report(resolved, fake_world):
    fake_world.state["fail_next"] = 2  # first two receiver calls fail
    rd = runner.run(resolved)
    p = rd.path
    for f in ("config.yaml", "environment.json", "events.jsonl", "plan.json", "trials.parquet",
              "metrics/summary.json", "tables/detections.csv"):  # fmt: skip
        assert (p / f).exists(), f

    df = pd.read_parquet(p / "trials.parquet")
    assert len(df) == 1 * 2 * 4  # sources x tasks x conditions
    assert (df["status"] == "failed").sum() == 2 and len(rd.list("failures")) == 2
    assert len(rd.list("generated")) == 2 and len(rd.list("detections")) == 2

    det = {d.stem: json.loads(d.read_text()) for d in rd.list("detections")}
    wm_doc = next(v for v in det.values() if v["watermark_enabled"])
    plain_doc = next(v for v in det.values() if not v["watermark_enabled"])
    assert (
        wm_doc["token_ids"]["watermark_detected"]
        and not plain_doc["token_ids"]["watermark_detected"]
    )

    # Leakage: C0/C1 requests never contain detector information.
    by_trial = {
        json.loads(r.read_text())["trial_id"]: json.loads(r.read_text())
        for r in rd.list("responses")
    }
    for _, row in df[df.status == "ok"].iterrows():
        sys_msg = by_trial[row.trial_id]["request"]["messages"][0]["content"]
        if row.condition in ("C0", "C1"):
            assert "EXPERIMENT METADATA" not in sys_msg and not row.instruction_applied
        else:
            assert "EXPERIMENT METADATA" in sys_msg and row.instruction_applied
    ok = df[df.status == "ok"]
    assert ok[ok.condition == "C3"]["metrics.marker_present"].all()
    assert not ok[ok.condition.isin(["C0", "C1", "C2"])]["metrics.marker_present"].any()

    # Resume: only the failed trials are retried, nothing is duplicated.
    n_sent = len(fake_world.sent)
    rd2 = runner.run(run_id=rd.run_id, root=resolved.settings.output_root)
    assert len(fake_world.sent) == n_sent + 2
    df2 = pd.read_parquet(rd2.path / "trials.parquet")
    assert (df2["status"] == "ok").all() and len(df2) == len(df)
    assert len(rd2.list("responses")) == 8 and len(rd2.list("generated")) == 2
    runner.run(run_id=rd.run_id, root=resolved.settings.output_root)  # idempotent
    assert len(fake_world.sent) == n_sent + 2

    # Traceability: every row points at real artifacts.
    for _, row in df2.iterrows():
        for col in ("source_artifact", "generated_artifact", "response_artifact"):
            assert (p / row[col]).exists()

    events = read_events(p / "events.jsonl")
    names = {e["event"] for e in events}
    for required in ("experiment.started", "document.loaded", "generation.started", "generation.completed",
                     "watermark.detection.started", "watermark.detection.completed",
                     "receiver.inference.started", "receiver.inference.completed", "receiver.inference.failed",
                     "evaluation.started", "evaluation.completed", "experiment.completed"):  # fmt: skip
        assert required in names, required

    from wm_eval.analysis.report import build_report

    report = build_report(p).read_text()
    assert "Watermark detection" in report and "Paired contrasts" in report
    assert (p / "figures" / "detection.png").exists() and (
        p / "tables" / "comparisons.csv"
    ).exists()

    # No secret or derived key anywhere in the run directory.
    keys = [str(k) for k in derive_keys(TEST_SECRET, 30)]
    for f in p.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".jsonl", ".yaml", ".csv", ".md"):
            text = f.read_text()
            assert TEST_SECRET not in text, f
            assert not any(f"[{k}," in text or f" {k}," in text for k in keys[:3]), f


def test_experiment_failure_is_recorded(resolved, fake_world, monkeypatch):
    monkeypatch.setattr(
        runner, "stage_evaluate", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        runner.run(resolved)
    run_dir = next(Path(resolved.settings.output_root).iterdir())
    events = read_events(run_dir / "events.jsonl")
    assert events[-1]["event"] == "experiment.failed" and "boom" in events[-1]["data"]["error"]


def test_stages_can_run_separately(resolved, fake_world):
    rd = runner.run(resolved, stages=("generate", "detect"))
    assert len(rd.list("generated")) == 2 and not rd.list("responses")
    runner.run(run_id=rd.run_id, root=resolved.settings.output_root)
    assert len(rd.list("responses")) == 8


def test_detector_only_run_without_receivers(resolved, fake_world):
    cfg = resolved.model_copy(deep=True)
    cfg.experiment = cfg.experiment.model_copy(
        update={
            "receivers": [],
            "conditions": ["C0", "C1"],
            "instrumentation": None,
            "generation_repeats": 3,
        }
    )
    cfg.receivers = {}
    rd = runner.run(cfg)
    assert len(rd.list("generated")) == 6 and len(rd.list("detections")) == 6
    assert not fake_world.sent and len(pd.read_parquet(rd.path / "trials.parquet")) == 0
    det = pd.read_parquet(rd.path / "tables" / "detections.parquet")
    assert set(det["generator"]) == {"tiny"}

    from wm_eval.analysis.report import build_report

    assert "TPR" in build_report(rd.path).read_text()


def test_evaluator_errors_are_retried_on_resume(resolved, fake_world, monkeypatch):
    from wm_eval.evaluation import verbosity

    real = verbosity.evaluate
    monkeypatch.setattr(
        verbosity, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA OOM"))
    )
    rd = runner.run(resolved)
    errored = [p for p in rd.list("evaluations") if json.loads(p.read_text())["errors"]]
    assert errored
    monkeypatch.setattr(verbosity, "evaluate", real)
    runner.run(run_id=rd.run_id, root=resolved.settings.output_root)
    assert not [p for p in rd.list("evaluations") if json.loads(p.read_text())["errors"]]
    assert len([p for p in rd.list("failures") if p.name.startswith("eval__")]) == len(errored)
    df = pd.read_parquet(rd.path / "trials.parquet")
    assert df["metrics.response_words"].notna().all()
