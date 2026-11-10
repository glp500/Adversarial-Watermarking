"""Runs the configured evaluators. Each evaluator module is independent; this file only
decides which text each one is applied to. One evaluator failing does not stop others."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from typing import Any

from wm_eval.schemas import EvaluationResult, SourceDocument

# Only these are loaded on demand, so an unused evaluator never downloads its model.
_MODULES = ("sentiment", "emotion", "contradiction", "similarity", "verbosity", "task_success")


def _mod(name: str):
    assert name in _MODULES, name
    return importlib.import_module(f"wm_eval.evaluation.{name}")


def _version(name: str) -> str:
    if name == "agent_deviation":
        return "agent_deviation/1"
    m = _mod(name)
    return m.version() if hasattr(m, "version") else m.VERSION


def _run(
    evaluators: list[str], steps: dict[str, Callable[[], dict[str, Any]]], key: str
) -> EvaluationResult:
    t0 = time.perf_counter()
    metrics: dict[str, Any] = {}
    versions: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name in evaluators:
        if name not in steps:
            continue
        try:
            metrics.update(steps[name]())
            versions[name] = _version(name)
        except Exception as e:  # recorded, never silently dropped
            errors[name] = f"{type(e).__name__}: {e}"
    return EvaluationResult(
        trial_id=key, metrics=metrics, evaluator_versions=versions,
        duration_s=time.perf_counter() - t0, errors=errors,
    )  # fmt: skip


def evaluate_response(
    evaluators: list[str],
    trial_id: str,
    task: str,
    response_text: str | None,
    document_text: str,
    source: SourceDocument,
    target_words: int | None,
    completion_tokens: int | None,
    marker: str | None,
    workspace_files: dict[str, str] | None = None,
    tool_trace: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    # For agent tasks the substantive output is the file the agent wrote.
    text = (
        (workspace_files or {}).get("summary.txt", "")
        if task == "agent_summarize"
        else response_text
    )
    if marker and text:
        text = text.replace(marker, "").strip()
    steps = {
        "verbosity": lambda: _mod("verbosity").evaluate(text, target_words, completion_tokens),
        "task_success": lambda: _mod("task_success").evaluate(
            task, response_text, source, marker, workspace_files
        ),
        "sentiment": lambda: _mod("sentiment").evaluate(text, "response_"),
        "emotion": lambda: _mod("emotion").evaluate(text, "response_"),
        "contradiction": lambda: {
            **_mod("contradiction").vs_source(text, source.text, "resp_vs_source_"),
            **_mod("contradiction").vs_source(text, document_text, "resp_vs_doc_"),
            **_mod("contradiction").internal(text),
        },
        "similarity": lambda: {
            "resp_vs_source_similarity": _mod("similarity").cosine(text, source.text)
        },
    }
    if tool_trace is not None:
        # Deviation metrics come from the tool trace only, independent of task success.
        from wm_eval.agent.simulated import deviation_metrics

        evaluators = [*evaluators, "agent_deviation"]
        steps["agent_deviation"] = lambda: deviation_metrics(tool_trace)
    return _run(evaluators, steps, trial_id)


def evaluate_document(
    evaluators: list[str], doc_id: str, text: str, source: SourceDocument
) -> EvaluationResult:
    """Source fidelity of a generated document (independent of any receiver)."""
    steps = {
        "verbosity": lambda: {"doc_words": _mod("verbosity").evaluate(text)["response_words"]},
        "sentiment": lambda: _mod("sentiment").evaluate(text, "doc_"),
        "emotion": lambda: _mod("emotion").evaluate(text, "doc_"),
        "contradiction": lambda: _mod("contradiction").vs_source(
            text, source.text, "doc_vs_source_"
        ),
        "similarity": lambda: {
            "doc_vs_source_similarity": _mod("similarity").cosine(text, source.text)
        },
    }
    return _run(evaluators, steps, doc_id)


def evaluate_source(evaluators: list[str], source: SourceDocument) -> EvaluationResult:
    steps = {
        "sentiment": lambda: _mod("sentiment").evaluate(source.text, "source_"),
        "emotion": lambda: _mod("emotion").evaluate(source.text, "source_"),
    }
    return _run(evaluators, steps, source.source_id)


def evaluate_pair(
    evaluators: list[str], key: str, text_a: str | None, text_b: str | None
) -> EvaluationResult:
    """Consistency between paired texts (watermarked vs unwatermarked)."""
    steps = {"similarity": lambda: {"pair_similarity": _mod("similarity").cosine(text_a, text_b)}}
    return _run(evaluators, steps, key)
