"""Shared helpers for Hugging Face classifier evaluators (lazy, cached, CPU by default).

Set WM_EVAL_EVAL_DEVICE=cuda to run evaluators on GPU (default cpu, because the GPU is
usually occupied by llama-server during a run).
"""

from __future__ import annotations

import os
from functools import cache

from wm_eval.data.preprocess import chunk_sentences, sentences, words


def device() -> str:
    return os.environ.get("WM_EVAL_EVAL_DEVICE", "cpu")


# Pinned Hugging Face revisions: evaluator outputs are only comparable for a fixed commit.
REVISIONS = {
    "siebert/sentiment-roberta-large-english": "74cea614e245b0832c770ec9aa51bd58df965b9c",
    "SamLowe/roberta-base-go_emotions": "d75048347613a25d77de8cf6412eaae9fa7b26be",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
# Repos whose pinned commit has only pytorch_model.bin. Without this, transformers silently
# loads safetensors from a bot conversion PR, i.e. weights not from the pinned commit.
BIN_ONLY = {"siebert/sentiment-roberta-large-english"}


@cache
def text_classifier(model_id: str):
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=model_id,
        revision=REVISIONS[model_id],
        top_k=None,
        device=device(),
        model_kwargs={"use_safetensors": False} if model_id in BIN_ONLY else {},
    )


def model_version(model_id: str) -> str:
    return f"{model_id}@{REVISIONS[model_id]}"


def chunked_label_scores(model_id: str, text: str, max_words: int = 200) -> dict[str, float] | None:
    """Per-label scores averaged over sentence chunks, weighted by chunk word count.
    Returns None for empty text."""
    chunks = chunk_sentences(sentences(text), max_words=max_words)
    if not chunks:
        return None
    pipe = text_classifier(model_id)
    results = pipe(chunks, truncation=True, max_length=512)
    weights = [max(len(words(c)), 1) for c in chunks]
    total = sum(weights)
    agg: dict[str, float] = {}
    for res, w in zip(results, weights, strict=True):
        for item in res:
            agg[item["label"].lower()] = (
                agg.get(item["label"].lower(), 0.0) + item["score"] * w / total
            )
    return agg
