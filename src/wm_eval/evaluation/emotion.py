"""Emotion classifier scores (SamLowe/roberta-base-go_emotions, 28 multi-label classes).
These are classifier scores for text, not measurements of anyone's emotions."""

from __future__ import annotations

from wm_eval.evaluation._hf import chunked_label_scores, model_version

MODEL_ID = "SamLowe/roberta-base-go_emotions"
REPORTED = ("anger", "fear", "sadness", "confusion", "neutral", "joy", "optimism", "nervousness")


def version() -> str:
    return model_version(MODEL_ID)


def evaluate(text: str | None, prefix: str = "") -> dict[str, float | None]:
    scores = chunked_label_scores(MODEL_ID, text or "")
    return {f"{prefix}emotion_{k}": scores.get(k) if scores else None for k in REPORTED}
