"""Paired comparisons with source-document clustering.

Unit of analysis: trials are paired by `pair_key` (same source, generator, generation
repeat, receiver, task, receiver repeat and receiver seed) and differ only in condition.
Repeated trials from one source document are correlated, so uncertainty is computed by
resampling *source documents* (cluster bootstrap) and tested with an exact/Monte-Carlo
sign-flip test on per-source mean differences. Nothing here is a causal claim beyond
the randomized/paired condition contrast itself.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CONTRASTS = (
    ("C1", "C0", "watermark effect (receiver not told)"),
    ("C3", "C2", "watermark effect (instrumented receiver)"),
    ("C2", "C0", "exposure effect (unwatermarked docs)"),
    ("C3", "C1", "exposure effect (watermarked docs)"),
)
CLUSTER = "source_id"


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric/bool metric columns that can differ between conditions."""
    cols = []
    for c in df.columns:
        if not c.startswith("metrics.") or c.startswith("metrics.source_"):
            continue
        if c.startswith("metrics.doc_pair") or c == "metrics.paired_response_similarity":
            continue  # already a within-pair quantity
        s = pd.to_numeric(
            df[c].map(lambda v: float(v) if isinstance(v, bool) else v), errors="coerce"
        )
        if s.notna().any():
            cols.append(c)
    return cols


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.map(lambda v: float(v) if isinstance(v, bool) else v), errors="coerce")


DOC_LEVEL = ("metrics.doc_",)  # properties of the generated document, not of a response


def paired_differences(df: pd.DataFrame, metric: str, treated: str, control: str) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    if metric.startswith(DOC_LEVEL):
        # One value per document: pair documents, not trials, so n is not inflated by
        # the number of tasks/receivers/repeats that reused the same document.
        doc_key = ["source_id", "generator", "generation_repeat"]
        ok = ok.drop_duplicates([*doc_key, "condition"]).assign(
            pair_key=lambda d: d[doc_key].astype(str).agg("|".join, axis=1)
        )
    a = ok[ok["condition"] == treated][["pair_key", CLUSTER, metric]]
    b = ok[ok["condition"] == control][["pair_key", metric]]
    m = a.merge(b, on="pair_key", suffixes=("_t", "_c"))
    m["diff"] = _numeric(m[f"{metric}_t"]) - _numeric(m[f"{metric}_c"])
    return m.dropna(subset=["diff"])[["pair_key", CLUSTER, "diff"]]


def cluster_bootstrap_ci(
    values: pd.Series, clusters: pd.Series, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI for the mean, resampling whole clusters with replacement."""
    groups = [v.to_numpy() for _, v in values.groupby(clusters.to_numpy())]
    if len(groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        stats[i] = np.concatenate([groups[j] for j in pick]).mean()
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def signflip_p(cluster_means: np.ndarray, n_mc: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip test of mean(cluster_means) == 0. Exact for <= 14 clusters."""
    k = len(cluster_means)
    if k == 0:
        return float("nan")
    obs = abs(cluster_means.mean())
    if k <= 14:
        signs = np.array(list(itertools.product((-1, 1), repeat=k)))
    else:
        signs = np.random.default_rng(seed).choice((-1, 1), size=(n_mc, k))
    null = np.abs((signs * cluster_means).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def compare(
    df: pd.DataFrame, metrics: list[str] | None = None, group_by: list[str] | None = None
) -> pd.DataFrame:
    metrics = metrics or metric_columns(df)
    conds = set(df["condition"])
    groups = [((), df)] if not group_by else list(df.groupby(group_by))
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for treated, control, label in CONTRASTS:
            if not {treated, control} <= conds:
                continue
            for m in metrics:
                d = paired_differences(g, m, treated, control)
                if d.empty:
                    continue
                cm = d.groupby(CLUSTER)["diff"].mean().to_numpy()
                lo, hi = cluster_bootstrap_ci(d["diff"], d[CLUSTER])
                rows.append({
                    **dict(zip(group_by or [], key, strict=False)),
                    "contrast": f"{treated}-{control}", "label": label, "metric": m.removeprefix("metrics."),
                    "n_pairs": len(d), "n_sources": d[CLUSTER].nunique(),
                    "mean_diff": d["diff"].mean(), "median_diff": d["diff"].median(),
                    "ci95_low": lo, "ci95_high": hi, "p_signflip": signflip_p(cm),
                })  # fmt: skip
    return pd.DataFrame(rows)


def condition_means(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or metric_columns(df)
    ok = df[df["status"] == "ok"]
    out = {
        m.removeprefix("metrics."): ok.groupby("condition")[m].apply(lambda s: _numeric(s).mean())
        for m in metrics
    }
    return pd.DataFrame(out)


def proportion(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"rate": None, "k": 0, "n": 0, "ci95": [None, None]}
    ci = binomtest(k, n).proportion_ci(method="wilson")
    return {"rate": k / n, "k": k, "n": n, "ci95": [ci.low, ci.high]}


def detection_rates(det: pd.DataFrame) -> dict[str, Any]:
    """TPR on watermarked docs and FPR on unwatermarked docs, per detector input kind."""
    out = {}
    for kind, g in det.groupby("input_kind"):
        pos, neg = g[g["watermark_enabled"]], g[~g["watermark_enabled"]]
        out[kind] = {
            "threshold": float(g["threshold"].iloc[0]),
            "tpr": proportion(int(pos["watermark_detected"].sum()), len(pos)),
            "fpr": proportion(int(neg["watermark_detected"].sum()), len(neg)),
            "median_p_watermarked": float(pos["p_value"].median()) if len(pos) else None,
            "median_p_unwatermarked": float(neg["p_value"].median()) if len(neg) else None,
        }
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, float | np.floating) and not np.isfinite(x):
        return None
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_summary(rd, df: pd.DataFrame) -> dict[str, Any]:
    """metrics/summary.json: completeness, detection rates, per-condition means."""
    det_path = Path(rd.path) / "tables" / "detections.parquet"
    det = pd.read_parquet(det_path) if det_path.exists() else pd.DataFrame()
    summary = {
        "run_id": rd.run_id,
        "n_trials_planned": len(df),
        "status_counts": df["status"].value_counts().to_dict(),
        "failed_attempts_total": int(df["n_failed_attempts"].sum()),
        "detection": detection_rates(det) if len(det) else None,
        "condition_means": condition_means(df).to_dict(orient="index")
        if (df["status"] == "ok").any()
        else {},
    }
    summary = _jsonable(summary)
    (Path(rd.path) / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
