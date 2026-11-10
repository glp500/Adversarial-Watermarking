import torch

from wm_eval.config import WatermarkConfig
from wm_eval.watermark.detector import detect, truncate_at_eos
from wm_eval.watermark.synthid import logits_processor, public_fields


def sample_tokens(
    cfg: WatermarkConfig, watermark: bool, n: int = 250, vocab: int = 1000, seed: int = 0
):
    """Sample from random logits, optionally through the SynthID processor (as generate does)."""
    proc = logits_processor(cfg)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, 4), generator=g)
    out = []
    for _ in range(n):
        scores = torch.randn(1, vocab, generator=g) * 2
        if watermark:
            scores = proc(ids, scores)
        tok = torch.multinomial(scores.softmax(-1), 1, generator=g)
        ids = torch.cat([ids, tok], dim=1)
        out.append(int(tok))
    return out


def test_watermarked_detected_unwatermarked_not(wm_cfg):
    wm = detect("a", sample_tokens(wm_cfg, True), wm_cfg, "tok")
    plain = detect("b", sample_tokens(wm_cfg, False), wm_cfg, "tok")
    assert wm.watermark_detected and wm.p_value < 1e-6 and wm.mean_g > 0.55
    assert not plain.watermark_detected and plain.p_value > wm_cfg.detection_threshold
    assert 0.4 < plain.mean_g < 0.6
    assert wm.method == "synthid_frequentist_binomial" and wm.key_id == "test_key"


def test_wrong_key_or_config_does_not_detect(wm_cfg, monkeypatch):
    tokens = sample_tokens(wm_cfg, True)
    monkeypatch.setenv("WM_SECRET", "a-different-secret")
    assert not detect("a", tokens, wm_cfg, "tok").watermark_detected
    monkeypatch.setenv("WM_SECRET", "unit-test-secret-do-not-use")
    other_ngram = wm_cfg.model_copy(update={"ngram_len": 4})
    assert not detect("a", tokens, other_ngram, "tok").watermark_detected


def test_false_positive_rate_is_controlled(wm_cfg):
    ps = [
        detect("x", sample_tokens(wm_cfg, False, n=120, seed=s), wm_cfg, "tok").p_value
        for s in range(40)
    ]
    assert sum(p < 0.01 for p in ps) <= 2  # expected 0.4 false positives at alpha=0.01


def test_short_and_empty_inputs(wm_cfg):
    r = detect("x", [5, 6], wm_cfg, "tok")
    assert r.p_value == 1.0 and not r.watermark_detected and r.n_scored == 0
    assert detect("x", [], wm_cfg, "tok").n_tokens == 0


def test_repeated_context_masked(wm_cfg):
    r = detect("x", [7, 8, 9, 10] * 50, wm_cfg, "tok")
    # Only the first occurrence of each 4-token context is scored.
    assert r.n_scored == 4 * wm_cfg.depth


def test_truncate_at_eos():
    assert truncate_at_eos([1, 2, 99, 3], [99]) == [1, 2]
    assert truncate_at_eos([1, 2], [99]) == [1, 2]


def test_detection_result_serialises_without_keys(wm_cfg):
    r = detect("a", sample_tokens(wm_cfg, True, n=60), wm_cfg, "tok")
    dumped = r.model_dump_json()
    for k in wm_cfg.keys()[:5]:
        assert f"{k}," not in dumped
    assert "secret" not in str(public_fields(wm_cfg))
