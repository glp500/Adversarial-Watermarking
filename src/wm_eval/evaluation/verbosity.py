"""Length and redundancy statistics. Pure Python, no models."""

from __future__ import annotations

from wm_eval.data.preprocess import sentences, words

VERSION = "verbosity/1"


def redundancy(text: str, n: int = 3) -> float | None:
    """1 - distinct/total word n-grams (0 = no repeated n-grams)."""
    w = [x.lower() for x in words(text)]
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else None


def evaluate(
    text: str | None, target_words: int | None = None, completion_tokens: int | None = None
) -> dict[str, float | int | None]:
    text = text or ""
    n_words = len(words(text))
    return {
        "response_words": n_words,
        "response_sentences": len(sentences(text)),
        "response_chars": len(text),
        "response_tokens": completion_tokens,
        "length_ratio": n_words / target_words if target_words else None,
        "redundancy": redundancy(text),
    }
