"""Check classifier evaluators against a small hand-labelled set (data/annotations/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

EMOTION_LABELS = ("anger", "fear", "sadness", "confusion", "neutral")


def _score(
    kind: str, gold: list[Any], pred: list[Any], items: list[dict], version: str
) -> dict[str, Any]:
    labels = sorted({*map(str, gold), *map(str, pred)})
    g, p = list(map(str, gold)), list(map(str, pred))
    return {
        "kind": kind,
        "evaluator_version": version,
        "n": len(g),
        "accuracy": accuracy_score(g, p),
        "macro_f1": f1_score(g, p, average="macro", zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(g, p, labels=labels).tolist(),
        "errors": [
            {**it, "predicted": pr} for it, gl, pr in zip(items, g, p, strict=True) if gl != pr
        ],
    }


def validate(path: str | Path) -> list[dict[str, Any]]:
    from wm_eval.evaluation import contradiction, emotion, sentiment
    from wm_eval.evaluation._hf import chunked_label_scores

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    out = []
    if items := by_kind.get("sentiment"):
        pred = []
        for it in items:
            s = sentiment.evaluate(it["text"])
            pred.append(
                "positive" if s["sentiment_positive"] >= s["sentiment_negative"] else "negative"
            )
        out.append(
            _score("sentiment", [i["label"] for i in items], pred, items, sentiment.version())
        )
    if items := by_kind.get("emotion"):
        pred = []
        for it in items:
            scores = chunked_label_scores(emotion.MODEL_ID, it["text"]) or {}
            pred.append(max(EMOTION_LABELS, key=lambda k: scores.get(k, 0.0)))
        out.append(_score("emotion", [i["label"] for i in items], pred, items, emotion.version()))
    if items := by_kind.get("contradiction"):
        probs = contradiction.nli_probs([(i["premise"], i["hypothesis"]) for i in items])
        pred = [p["contradiction"] > 0.5 for p in probs]
        out.append(
            _score(
                "contradiction", [i["label"] for i in items], pred, items, contradiction.version()
            )
        )
    return out
