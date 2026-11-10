"""Shared fixtures. Nothing here downloads a model."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wm_eval.config import (
    ExperimentConfig,
    GeneratorSpec,
    ReceiverSpec,
    ResolvedConfig,
    Settings,
    WatermarkConfig,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_SECRET = "unit-test-secret-do-not-use"


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("WM_SECRET", TEST_SECRET)


@pytest.fixture
def wm_cfg() -> WatermarkConfig:
    return WatermarkConfig(config_id="test_wm", key_id="test_key", depth=30, ngram_len=5)


@pytest.fixture
def tiny_lm():
    """A 1-layer random GPT-2 with a word-level tokenizer, built in memory."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[EOS]": 1, **{f"w{i}": i + 2 for i in range(800)}}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(vocab), n_positions=1024, n_embd=32, n_layer=1, n_head=2,
                     bos_token_id=1, eos_token_id=1)  # fmt: skip
    model = GPT2LMHeadModel(cfg).eval()
    model.generation_config.eos_token_id = 1
    return model, tokenizer


@pytest.fixture
def resolved(tmp_path, wm_cfg) -> ResolvedConfig:
    exp = ExperimentConfig(
        name="test",
        seed=7,
        sources=str(ROOT / "tests" / "fixtures" / "articles.jsonl"),
        generators=["tiny"],
        receivers=["fake"],
        watermark="test_wm",
        conditions=["C0", "C1", "C2", "C3"],
        tasks=["summarize", "json"],
        evaluators=["verbosity", "task_success"],
        generation={"max_new_tokens": 120, "temperature": 1.0, "top_k": 40, "top_p": 1.0},
        rewrite_prompt="w1 w2 w3 {article}",
        instrumentation={"mode": "instruction"},
    )
    return ResolvedConfig(
        settings=Settings(output_root=str(tmp_path / "results")),
        experiment=exp,
        generators={"tiny": GeneratorSpec(hf_id="tiny/model", device="cpu", dtype="float32")},
        receivers={"fake": ReceiverSpec(base_url="http://127.0.0.1:9/v1", model="fake")},
        watermark=wm_cfg,
    )


def pytest_collection_modifyitems(config, items):
    run_model = os.environ.get("WM_EVAL_MODEL_TESTS") == "1"
    run_server = os.environ.get("WM_EVAL_SERVER_TESTS") == "1"
    for item in items:
        if "model" in item.keywords and not run_model:
            item.add_marker(
                pytest.mark.skip(reason="set WM_EVAL_MODEL_TESTS=1 to run real-model tests")
            )
        if "server" in item.keywords and not run_server:
            item.add_marker(
                pytest.mark.skip(reason="set WM_EVAL_SERVER_TESTS=1 (needs llama-server)")
            )
