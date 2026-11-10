"""Load a Transformers generator. The class is configured per model (Gemma 4 and Qwen3.5
are multimodal `*ForConditionalGeneration` checkpoints), so nothing assumes
AutoModelForCausalLM."""

from __future__ import annotations

import gc
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

from wm_eval.config import GeneratorSpec

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_tokenizer(spec: GeneratorSpec):
    return AutoTokenizer.from_pretrained(spec.tokenizer_id or spec.hf_id, revision=spec.revision)


def load_model(spec: GeneratorSpec) -> tuple[Any, Any]:
    cls = getattr(transformers, spec.model_class)
    kwargs: dict[str, Any] = {"revision": spec.revision, "dtype": DTYPES[spec.dtype]}
    if spec.device == "auto":
        kwargs["device_map"] = "auto"
    model = cls.from_pretrained(spec.hf_id, **kwargs)
    if spec.device != "auto":
        model = model.to(spec.device)
    model.eval()
    return model, load_tokenizer(spec)


def model_metadata(spec: GeneratorSpec, model: Any) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        **spec.model_dump(),
        "tokenizer_id": spec.tokenizer_id or spec.hf_id,
        "architectures": getattr(cfg, "architectures", None),
        "model_class_loaded": type(model).__name__,
        "device_map": getattr(model, "hf_device_map", None) or str(getattr(model, "device", "")),
    }


def free_memory() -> None:
    """Call after the caller has dropped its references to the model (`del model`)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
