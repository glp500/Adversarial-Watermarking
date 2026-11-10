"""Sentiment classifier scores (siebert/sentiment-roberta-large-english).
Classifier outputs, not ground truth; validate against manual labels before use."""

from __future__ import annotations

from wm_eval.evaluation._hf import chunked_label_scores, model_version

MODEL_ID = "siebert/sentiment-roberta-large-english"


def version() -> str:
    return model_version(MODEL_ID)


def evaluate(text: str | None, prefix: str = "") -> dict[str, float | None]:
    scores = chunked_label_scores(MODEL_ID, text or "")
    if scores is None:
        return {f"{prefix}sentiment_positive": None, f"{prefix}sentiment_negative": None}
    return {
        f"{prefix}sentiment_positive": scores.get("positive"),
        f"{prefix}sentiment_negative": scores.get("negative"),
    }
