"""Optional tests against real checkpoints / a running llama-server. Skipped by default.

WM_EVAL_MODEL_TESTS=1 pytest -m model      # downloads/uses Qwen/Qwen3.5-2B
WM_EVAL_SERVER_TESTS=1 pytest -m server    # needs llama-server on the configured port
"""

import pytest

from tests.conftest import ROOT
from wm_eval.config import GenerationParams, ReceiverParams, load_config
from wm_eval.schemas import GenerationRequest

CFG = load_config(ROOT / "configs" / "experiments" / "smoke.yaml")


@pytest.mark.model
def test_qwen_watermark_roundtrip():
    from wm_eval.generation.generator import generate
    from wm_eval.generation.models import load_model, model_metadata
    from wm_eval.watermark.detector import detect, detect_text

    spec = CFG.generators["qwen3.5-2b"]
    model, tok = load_model(spec)
    params = GenerationParams(max_new_tokens=200)
    out = {}
    for wm in (True, False):
        req = GenerationRequest(source_id="s", prompt="Write a short story about a lighthouse keeper.",
                                generator="qwen3.5-2b", watermark_enabled=wm, seed=1)  # fmt: skip
        out[wm] = generate(
            model,
            tok,
            req,
            f"d{wm}",
            "test",
            spec,
            params,
            CFG.watermark,
            model_metadata(spec, model),
        )
    assert detect(
        "w", out[True].token_ids, CFG.watermark, spec.hf_id, out[True].metadata["eos_token_ids"]
    ).watermark_detected
    assert not detect(
        "n", out[False].token_ids, CFG.watermark, spec.hf_id, out[False].metadata["eos_token_ids"]
    ).watermark_detected
    assert detect_text("w", out[True].text, tok, CFG.watermark, spec.hf_id).p_value < 0.05


@pytest.mark.server
def test_llama_server_roundtrip():
    from wm_eval.receiver.inference import build_messages, run_receiver
    from wm_eval.receiver.llama_client import LlamaClient

    name, spec = next(iter(CFG.receivers.items()))
    client = LlamaClient(spec)
    assert client.health()["ok"]
    res = run_receiver(client, "t", "r", name, build_messages("Summarize in one sentence.", "The sky is blue."),
                       ReceiverParams(max_tokens=60), seed=1)  # fmt: skip
    assert res.status == "ok" and res.text and res.completion_tokens
