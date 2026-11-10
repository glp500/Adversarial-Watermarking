"""Frequentist SynthID detector.

Under H0 (text not generated with this key) every unmasked g-value is a Bernoulli draw
with success rate p0 = mean of the SynthID sampling table (~0.5). Watermarking pushes
g-values up, so we use an exact one-sided binomial test on the count of g == 1.

* The p-value is P(score >= observed | H0). It is NOT the probability that a text is
  watermarked.
* Masked out: the first ngram_len-1 tokens (no full context), tokens at/after EOS, and
  tokens whose context already appeared earlier (repeated contexts are not independent
  and were not watermarked during generation).
* Detection must use the generator's own tokenizer and the same WatermarkConfig.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
from scipy.stats import binomtest

from wm_eval.config import WatermarkConfig
from wm_eval.schemas import DetectionResult
from wm_eval.watermark.synthid import logits_processor

METHOD = "synthid_frequentist_binomial"


def truncate_at_eos(token_ids: Sequence[int], eos_ids: Sequence[int]) -> list[int]:
    eos = set(eos_ids)
    for i, t in enumerate(token_ids):
        if t in eos:
            return list(token_ids[:i])
    return list(token_ids)


def score_tokens(token_ids: Sequence[int], cfg: WatermarkConfig) -> tuple[int, int, float]:
    """Return (n_ones, n_scored, p0) from unmasked g-values."""
    proc = logits_processor(cfg)
    p0 = proc.sampling_table.float().mean().item()
    if len(token_ids) < cfg.ngram_len:
        return 0, 0, p0
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    g = proc.compute_g_values(ids)  # [1, T - ngram_len + 1, depth]
    mask = proc.compute_context_repetition_mask(ids).bool()  # [1, T - ngram_len + 1]
    g = g[mask]  # [n_unmasked, depth]
    return int(g.sum().item()), int(g.numel()), p0


def detect(
    doc_id: str,
    token_ids: Sequence[int],
    cfg: WatermarkConfig,
    tokenizer_id: str,
    eos_ids: Sequence[int] = (),
    input_kind: str = "token_ids",
) -> DetectionResult:
    t0 = time.perf_counter()
    ids = truncate_at_eos(token_ids, eos_ids)
    n_ones, n, p0 = score_tokens(ids, cfg)
    p_value = binomtest(n_ones, n, p0, alternative="greater").pvalue if n else 1.0
    return DetectionResult(
        doc_id=doc_id,
        method=METHOD,
        input_kind=input_kind,
        watermark_config_id=cfg.config_id,
        key_id=cfg.key_id,
        tokenizer_id=tokenizer_id,
        watermark_detected=bool(n and p_value < cfg.detection_threshold),
        p_value=float(p_value),
        threshold=cfg.detection_threshold,
        mean_g=n_ones / n if n else float("nan"),
        n_scored=n,
        n_tokens=len(ids),
        duration_s=time.perf_counter() - t0,
    )


def detect_text(
    doc_id: str, text: str, tokenizer, cfg: WatermarkConfig, tokenizer_id: str
) -> DetectionResult:
    """Retokenize with the *generator's* tokenizer and detect. Never use a receiver tokenizer."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return detect(doc_id, ids, cfg, tokenizer_id, input_kind="retokenized_text")
