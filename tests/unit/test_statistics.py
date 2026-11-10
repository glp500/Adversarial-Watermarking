import numpy as np
import pandas as pd

from wm_eval.analysis.statistics import (
    cluster_bootstrap_ci,
    compare,
    detection_rates,
    paired_differences,
    signflip_p,
)


def trials_df(effect: float, n_sources: int = 6, reps: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sources):
        base = rng.normal(100, 20)  # source-level random effect (correlated repetitions)
        for r in range(reps):
            pk = f"s{s}r{r}"
            noise = rng.normal(0, 1)
            rows.append(
                {
                    "pair_key": pk,
                    "source_id": f"s{s}",
                    "condition": "C0",
                    "status": "ok",
                    "metrics.response_words": base + noise,
                    "metrics.task_success": True,
                }
            )
            rows.append(
                {
                    "pair_key": pk,
                    "source_id": f"s{s}",
                    "condition": "C1",
                    "status": "ok",
                    "metrics.response_words": base + noise + effect,
                    "metrics.task_success": False,
                }
            )
    return pd.DataFrame(rows)


def test_paired_differences_recover_effect():
    d = paired_differences(trials_df(5.0), "metrics.response_words", "C1", "C0")
    assert len(d) == 24 and np.allclose(d["diff"], 5.0)


def test_compare_reports_ci_and_counts():
    out = compare(trials_df(5.0))
    row = out[(out.metric == "response_words") & (out.contrast == "C1-C0")].iloc[0]
    assert row.n_pairs == 24 and row.n_sources == 6
    assert row.ci95_low <= 5.0 <= row.ci95_high
    ts = out[out.metric == "task_success"].iloc[0]
    assert ts.mean_diff == -1.0


def test_failed_trials_excluded():
    df = trials_df(5.0)
    df.loc[0, "status"] = "failed"
    assert len(paired_differences(df, "metrics.response_words", "C1", "C0")) == 23


def test_cluster_bootstrap_wider_than_naive_for_correlated_data():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(5), 20)
    values = pd.Series(np.repeat(rng.normal(0, 1, 5), 20) + rng.normal(0, 0.1, 100))
    lo, hi = cluster_bootstrap_ci(values, pd.Series(clusters))
    naive = 1.96 * values.std() / np.sqrt(len(values)) * 2
    assert (hi - lo) > naive


def test_single_cluster_gives_nan_ci():
    lo, hi = cluster_bootstrap_ci(pd.Series([1.0, 2.0]), pd.Series(["a", "a"]))
    assert np.isnan(lo) and np.isnan(hi)


def test_signflip():
    assert signflip_p(np.array([1.0] * 8)) == 2 / 256
    assert signflip_p(np.array([1.0, -1.0])) == 1.0


def test_detection_rates():
    det = pd.DataFrame({"input_kind": ["token_ids"] * 4, "watermark_enabled": [True, True, False, False],
                        "watermark_detected": [True, False, False, False], "p_value": [1e-5, 0.2, 0.5, 0.9],
                        "threshold": [0.01] * 4})  # fmt: skip
    r = detection_rates(det)["token_ids"]
    assert r["tpr"]["rate"] == 0.5 and r["fpr"]["rate"] == 0.0 and r["fpr"]["ci95"][1] > 0


def test_doc_level_metrics_paired_once_per_document():
    df = trials_df(5.0, n_sources=2, reps=3)
    df["generator"], df["generation_repeat"] = "g", 0
    df["metrics.doc_words"] = np.where(df["condition"] == "C1", 110.0, 100.0)
    d = paired_differences(df, "metrics.doc_words", "C1", "C0")
    assert len(d) == 2 and (d["diff"] == 10).all()
