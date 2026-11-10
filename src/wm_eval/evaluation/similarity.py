"""Semantic similarity with sentence-transformers (cosine of mean chunk embeddings)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from wm_eval.data.preprocess import chunk_sentences, sentences
from wm_eval.evaluation._hf import REVISIONS, device, model_version

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, revision=REVISIONS[MODEL_ID], device=device())


def version() -> str:
    return model_version(MODEL_ID)


def embed(text: str) -> np.ndarray | None:
    chunks = chunk_sentences(sentences(text), max_words=150)
    if not chunks:
        return None
    v = _model().encode(chunks, normalize_embeddings=True).mean(axis=0)
    return v / np.linalg.norm(v)


def cosine(a: str | None, b: str | None) -> float | None:
    ea, eb = embed(a or ""), embed(b or "")
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))
