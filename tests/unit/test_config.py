import pytest
from pydantic import ValidationError

from tests.conftest import ROOT
from wm_eval.config import (
    ExperimentConfig,
    GenerationParams,
    WatermarkConfig,
    derive_keys,
    load_config,
    resolve,
)

BASE = dict(name="x", sources="s.jsonl", generators=["g"], receivers=["r"], watermark="w")


def test_all_shipped_experiment_configs_load():
    for path in (ROOT / "configs" / "experiments").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.experiment.name


@pytest.mark.parametrize(
    "bad",
    [
        {"tasks": ["not_a_task"]},
        {"evaluators": ["vibes"]},
        {"conditions": ["C9"]},
        {"conditions": ["C0", "C3"]},  # exposed condition without instrumentation
        {"generation_repeats": 0},
        {"unknown_field": 1},
    ],
)
def test_invalid_experiment_values_rejected(bad):
    with pytest.raises(ValidationError):
        ExperimentConfig(**{**BASE, **bad})


def test_exposed_conditions_need_detector():
    with pytest.raises(ValidationError):
        ExperimentConfig(**BASE, conditions=["C2"], instrumentation={}, run_detector=False)


@pytest.mark.parametrize("bad", [{"temperature": 0}, {"top_p": 1.5}, {"max_new_tokens": 0}])
def test_invalid_generation_params(bad):
    with pytest.raises(ValidationError):
        GenerationParams(**bad)


def test_invalid_watermark_params():
    with pytest.raises(ValidationError):
        WatermarkConfig(config_id="a", key_id="k", detection_threshold=1.5)
    with pytest.raises(ValidationError):
        WatermarkConfig(config_id="a", key_id="k", ngram_len=1)


def test_undefined_references_rejected(wm_cfg):
    from wm_eval.config import Settings

    exp = ExperimentConfig(**{**BASE, "watermark": "test_wm"})
    with pytest.raises(ValueError, match="undefined"):
        resolve(Settings(), exp, {}, {}, {"test_wm": wm_cfg})


def test_keys_derived_deterministically_and_need_secret(monkeypatch, wm_cfg):
    assert derive_keys("a", 5) == derive_keys("a", 5)
    assert derive_keys("a", 5) != derive_keys("b", 5)
    assert len(wm_cfg.keys()) == 30
    monkeypatch.delenv("WM_SECRET")
    with pytest.raises(RuntimeError, match="WM_SECRET"):
        wm_cfg.keys()
