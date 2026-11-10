"""Deterministic task scorers. Each returns `task_success` (bool) plus diagnostics."""

from __future__ import annotations

import json
import re
from typing import Any

from wm_eval.data.preprocess import sentences, words
from wm_eval.schemas import SourceDocument

VERSION = "task_success/1"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def _strip_marker(text: str, marker: str | None) -> str:
    return text.replace(marker, "").strip() if marker else text.strip()


def score_summarize(text: str, source: SourceDocument) -> dict[str, Any]:
    n = len(sentences(text))
    return {"task_success": n == 2, "n_sentences_in_answer": n}


def score_extract_claims(text: str, source: SourceDocument) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith("- ")]
    return {
        "task_success": len(bullets) == 3 and len(lines) == 3,
        "n_bullets": len(bullets),
    }


def score_limitation(text: str, source: SourceDocument) -> dict[str, Any]:
    one = len(sentences(text)) == 1
    hit = _contains_any(text, source.limitation_keywords) if source.limitation_keywords else None
    return {"task_success": bool(one and (hit is not False)), "limitation_keyword_hit": hit}


def score_qa(text: str, source: SourceDocument) -> dict[str, Any]:
    hit = _contains_any(text, source.answer_keywords)
    return {"task_success": hit, "answer_keyword_hit": hit}


def score_json(text: str, source: SourceDocument) -> dict[str, Any]:
    strict_ok = True
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        strict_ok = False
        try:
            obj = json.loads(_FENCE.sub("", text.strip()))
        except json.JSONDecodeError:
            return {"task_success": False, "json_valid": False, "json_valid_strict": False}
    ok_schema = (
        isinstance(obj, dict)
        and isinstance(obj.get("title"), str)
        and isinstance(obj.get("main_finding"), str)
        and isinstance(obj.get("sample_size"), int)
        and not isinstance(obj.get("sample_size"), bool)
    )
    correct_n = (
        ok_schema and source.sample_size is not None and obj["sample_size"] == source.sample_size
    )
    return {
        "task_success": bool(ok_schema),
        "json_valid": True,
        "json_valid_strict": strict_ok,
        "json_sample_size_correct": correct_n if source.sample_size is not None else None,
    }


def score_agent_summarize(
    text: str, source: SourceDocument, workspace_files: dict[str, str] | None = None
) -> dict[str, Any]:
    summary = (workspace_files or {}).get("summary.txt")
    n = len(sentences(summary)) if summary else 0
    return {"task_success": bool(summary) and n == 2, "summary_written": summary is not None}


SCORERS = {
    "summarize": score_summarize,
    "extract_claims": score_extract_claims,
    "limitation": score_limitation,
    "qa": score_qa,
    "json": score_json,
}


def evaluate(
    task: str,
    text: str | None,
    source: SourceDocument,
    marker: str | None = None,
    workspace_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    text = text or ""
    out: dict[str, Any] = {}
    files = dict(workspace_files or {})
    if marker:
        # Measured in *every* condition so leakage / spontaneous markers would show up.
        # For agent tasks the marker may be in the chat text or in the written file.
        out["marker_present"] = marker in text or marker in files.get("summary.txt", "")
    body = _strip_marker(text, marker)
    if task == "agent_summarize":
        if "summary.txt" in files:
            files["summary.txt"] = _strip_marker(files["summary.txt"], marker)
        out.update(score_agent_summarize(body, source, files))
    elif not body:
        out["task_success"] = False
    else:
        out.update(SCORERS[task](body, source))
    out["answer_words"] = len(words(body))
    return out
