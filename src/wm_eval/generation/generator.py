"""Generate a document with or without the SynthID watermark from the *same* checkpoint,
seed and sampling parameters. Only `watermarking_config` differs between treatments.

Note: an identical seed makes the two samples share RNG draws, so they are correlated,
but they are NOT guaranteed to be semantically identical.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
from transformers import GenerationConfig, set_seed

from wm_eval.config import GenerationParams, GeneratorSpec, WatermarkConfig
from wm_eval.schemas import GenerationRequest, GenerationResult
from wm_eval.watermark import synthid


def prompt_ids(tokenizer: Any, prompt: str, chat_template_kwargs: dict[str, Any]) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            **chat_template_kwargs,
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer(prompt)["input_ids"]


def eos_ids(model: Any, tokenizer: Any) -> list[int]:
    """Union of the checkpoint's and the tokenizer's EOS ids. Some snapshots lack
    generation_config.json (e.g. Qwen3.5-2B), leaving only <|endoftext|> and not the chat
    turn end <|im_end|> that the tokenizer declares."""
    ids: set[int] = set()
    for e in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if e is not None:
            ids.update([e] if isinstance(e, int) else e)
    return sorted(ids)


def sampling_config(model: Any, params: GenerationParams, eos: list[int]) -> GenerationConfig:
    """Start from the checkpoint's defaults (eos/pad ids) but pin every sampling knob."""
    gc = GenerationConfig(**model.generation_config.to_dict())
    gc.update(
        do_sample=True,
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        max_new_tokens=params.max_new_tokens,
        repetition_penalty=1.0,
        num_beams=1,
    )
    gc.eos_token_id = eos
    if gc.pad_token_id is None:
        gc.pad_token_id = eos[0] if eos else 0
    return gc


def generate(
    model: Any,
    tokenizer: Any,
    request: GenerationRequest,
    doc_id: str,
    run_id: str,
    spec: GeneratorSpec,
    params: GenerationParams,
    wm_cfg: WatermarkConfig,
    model_meta: dict[str, Any],
) -> GenerationResult:
    p_ids = prompt_ids(tokenizer, request.prompt, spec.chat_template_kwargs)
    device = next(model.parameters()).device
    input_ids = torch.tensor([p_ids], device=device)
    eos = eos_ids(model, tokenizer)
    gen_cfg = sampling_config(model, params, eos)
    # The watermark config (which holds the keys) goes on a private copy, so the recorded
    # generation config below never contains key material.
    run_cfg = copy.deepcopy(gen_cfg)
    if request.watermark_enabled:
        run_cfg.watermarking_config = synthid.generation_config(wm_cfg)

    set_seed(request.seed)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=run_cfg,
        )
    dt = time.perf_counter() - t0
    new_ids = out[0, len(p_ids) :].tolist()
    finished = any(t in eos for t in new_ids)
    return GenerationResult(
        doc_id=doc_id,
        run_id=run_id,
        source_id=request.source_id,
        generator=request.generator,
        watermark_enabled=request.watermark_enabled,
        watermark_config_id=wm_cfg.config_id if request.watermark_enabled else None,
        watermark_key_id=wm_cfg.key_id if request.watermark_enabled else None,
        seed=request.seed,
        prompt=request.prompt,
        prompt_token_ids=p_ids,
        token_ids=new_ids,
        text=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        finish_reason="eos" if finished else "length",
        duration_s=dt,
        metadata={
            "model": model_meta,
            "generation_params": params.model_dump(),
            "generation_config": gen_cfg.to_diff_dict(),
            "eos_token_ids": eos,
            # Watermark parameters (no keys) recorded for both arms, so the arms are
            # comparable; `watermark_enabled` says which arm actually applied it.
            "watermark_params": synthid.public_fields(wm_cfg),
        },
    )
