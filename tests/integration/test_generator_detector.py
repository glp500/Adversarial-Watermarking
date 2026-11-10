"""Real `model.generate` + SynthID on a tiny in-memory model, then detection."""

from wm_eval.config import GenerationParams, GeneratorSpec
from wm_eval.generation.generator import generate
from wm_eval.generation.models import model_metadata
from wm_eval.schemas import GenerationRequest
from wm_eval.watermark.detector import detect, detect_text

SPEC = GeneratorSpec(hf_id="tiny/model", device="cpu", dtype="float32")
PARAMS = GenerationParams(max_new_tokens=200, temperature=1.0, top_k=40)


def gen(tiny_lm, wm_cfg, watermark: bool, seed: int = 3):
    model, tok = tiny_lm
    req = GenerationRequest(source_id="s", prompt="w1 w2 w3 w4 w5", generator="tiny",
                            watermark_enabled=watermark, seed=seed)  # fmt: skip
    return generate(
        model,
        tok,
        req,
        f"d{int(watermark)}",
        "run",
        SPEC,
        PARAMS,
        wm_cfg,
        model_metadata(SPEC, model),
    )


def test_paired_generation_and_detection(tiny_lm, wm_cfg):
    wm, plain = gen(tiny_lm, wm_cfg, True), gen(tiny_lm, wm_cfg, False)
    assert (
        wm.watermark_enabled
        and wm.watermark_config_id == "test_wm"
        and plain.watermark_config_id is None
    )
    assert wm.prompt_token_ids == plain.prompt_token_ids  # same prompt, separate from output
    assert wm.metadata["generation_config"] == plain.metadata["generation_config"]  # same sampling
    assert wm.token_ids != plain.token_ids
    d_wm = detect(wm.doc_id, wm.token_ids, wm_cfg, "tiny", wm.metadata["eos_token_ids"])
    d_plain = detect(plain.doc_id, plain.token_ids, wm_cfg, "tiny", plain.metadata["eos_token_ids"])
    assert d_wm.watermark_detected and d_wm.p_value < 1e-4
    assert not d_plain.watermark_detected


def test_token_ids_recoverable_and_retokenization_matches(tiny_lm, wm_cfg):
    _, tok = tiny_lm
    wm = gen(tiny_lm, wm_cfg, True)
    assert tok.decode(wm.token_ids, skip_special_tokens=True).strip() == wm.text
    d_text = detect_text(wm.doc_id, wm.text, tok, wm_cfg, "tiny")
    assert d_text.input_kind == "retokenized_text" and d_text.watermark_detected


def test_same_seed_is_reproducible(tiny_lm, wm_cfg):
    assert (
        gen(tiny_lm, wm_cfg, True, seed=9).token_ids == gen(tiny_lm, wm_cfg, True, seed=9).token_ids
    )


def test_watermark_is_device_independent(wm_cfg):
    """Regression: transformers' default table differs between CPU and CUDA RNGs."""
    import pytest
    import torch

    from wm_eval.watermark.synthid import logits_processor

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cpu, cuda = logits_processor(wm_cfg, "cpu"), logits_processor(wm_cfg, "cuda")
    assert torch.equal(cpu.sampling_table, cuda.sampling_table.cpu())


def test_generate_on_gpu_detect_on_cpu(tiny_lm, wm_cfg):
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model, tok = tiny_lm
    model = model.to("cuda")
    wm = gen((model, tok), wm_cfg, True)
    assert detect(
        wm.doc_id, wm.token_ids, wm_cfg, "tiny", wm.metadata["eos_token_ids"]
    ).watermark_detected


def test_eos_ids_union_of_model_and_tokenizer():
    from types import SimpleNamespace

    from wm_eval.generation.generator import eos_ids

    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
    assert eos_ids(model, SimpleNamespace(eos_token_id=248046)) == [248044, 248046]
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 106]))
    assert eos_ids(model, SimpleNamespace(eos_token_id=1)) == [1, 106]
