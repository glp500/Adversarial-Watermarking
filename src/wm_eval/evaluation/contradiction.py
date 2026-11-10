"""NLI-based contradiction scores (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

Atomic-claim approach: the hypothesis text is split into sentence-level claims; each claim
is compared with short windows of the premise (source) rather than the whole document,
and the window most relevant to the claim (highest max(entail, contradict)) decides its
label. Internal contradiction compares response sentences pairwise (both directions).
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import torch

from wm_eval.data.preprocess import sentences, words
from wm_eval.evaluation._hf import REVISIONS, device, model_version

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_CLAIMS = 30
MAX_PAIRS = 120


@lru_cache(maxsize=1)
def _nli():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rev = REVISIONS[MODEL_ID]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=rev)
    model = (
        AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=rev)
        .to(device())
        .eval()
    )
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    return tok, model, labels


def version() -> str:
    return model_version(MODEL_ID)


def nli_probs(pairs: list[tuple[str, str]], batch_size: int = 16) -> list[dict[str, float]]:
    """[(premise, hypothesis)] -> [{'entailment', 'neutral', 'contradiction'}]."""
    tok, model, labels = _nli()
    out = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        enc = tok(
            [p for p, _ in batch], [h for _, h in batch],
            truncation=True, max_length=512, padding=True, return_tensors="pt",
        ).to(model.device)  # fmt: skip
        with torch.inference_mode():
            probs = model(**enc).logits.softmax(-1).cpu()
        out += [{k: float(row[j]) for k, j in labels.items()} for row in probs]
    return out


def claims(text: str) -> list[str]:
    sents = [s.lstrip("-*• ").strip() for s in sentences(text)]
    return [s for s in sents if len(words(s)) >= 4][:MAX_CLAIMS]


def windows(text: str, size: int = 3) -> list[str]:
    s = sentences(text)
    if len(s) <= size:
        return [" ".join(s)] if s else []
    return [" ".join(s[i : i + size]) for i in range(len(s) - size + 1)]


def vs_source(hypothesis: str | None, source: str, prefix: str) -> dict[str, float | int | None]:
    cl, wins = claims(hypothesis or ""), windows(source)
    if not cl or not wins:
        return {f"{prefix}contradiction_mean": None, f"{prefix}contradiction_rate": None,
                f"{prefix}entailment_mean": None, f"{prefix}n_claims": len(cl)}  # fmt: skip
    probs = nli_probs([(w, c) for c in cl for w in wins])
    contra, entail = [], []
    for i in range(len(cl)):
        rows = probs[i * len(wins) : (i + 1) * len(wins)]
        best = max(rows, key=lambda r: max(r["entailment"], r["contradiction"]))
        contra.append(best["contradiction"])
        entail.append(best["entailment"])
    return {
        f"{prefix}contradiction_mean": sum(contra) / len(contra),
        f"{prefix}contradiction_rate": sum(c > 0.5 for c in contra) / len(contra),
        f"{prefix}entailment_mean": sum(entail) / len(entail),
        f"{prefix}n_claims": len(cl),
    }


def internal(text: str | None) -> dict[str, float | None]:
    cl = claims(text or "")
    pairs = [(a, b) for a, b in itertools.permutations(cl, 2)][:MAX_PAIRS]
    if not pairs:
        return {"internal_contradiction_max": None, "internal_contradiction_rate": None}
    c = [p["contradiction"] for p in nli_probs(pairs)]
    return {
        "internal_contradiction_max": max(c),
        "internal_contradiction_rate": sum(x > 0.5 for x in c) / len(c),
    }
