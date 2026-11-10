"""`wm-eval report`: figures, comparison tables and a human-readable report from saved
results only (no models are run)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wm_eval.analysis.statistics import (  # noqa: E402
    _numeric,
    cluster_bootstrap_ci,
    compare,
    detection_rates,
    write_summary,
)
from wm_eval.observability.events import read_events  # noqa: E402

# One visual grammar for the 2x2 design, used by every figure:
#   colour = watermark (Okabe-Ito blue = none, vermilion = watermarked)
#   fill   = receiver told about detection (solid = not told, hatched/open = told)
NO_WM, WM = "#0072B2", "#D55E00"
COND = {
    "C0": {"color": NO_WM, "told": False, "label": "C0\nplain doc\nnot told"},
    "C1": {"color": WM, "told": False, "label": "C1\nwatermarked\nnot told"},
    "C2": {"color": NO_WM, "told": True, "label": "C2\nplain doc\ntold"},
    "C3": {"color": WM, "told": True, "label": "C3\nwatermarked\ntold"},
}
COND_ORDER = ["C0", "C1", "C2", "C3"]  # grouped so each watermark contrast sits side by side
COND_X = {"C0": 0, "C1": 1.15, "C2": 2.85, "C3": 4.0}
CONTRAST_LABEL = {
    "C1-C0": "C1 − C0  watermark, not told",
    "C3-C2": "C3 − C2  watermark, told",
    "C2-C0": "C2 − C0  telling, no watermark",
    "C3-C1": "C3 − C1  telling, watermarked",
}
METRIC_LABEL = {
    "response_words": "Response length (words)",
    "task_success": "Task success (rate)",
    "resp_vs_source_contradiction_rate": "Claims contradicting source (share)",
    "response_sentiment_negative": "Negative sentiment (probability)",
}
RC = {"figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5, "axes.titlesize": 9,
      "axes.titleweight": "bold", "axes.labelsize": 8.5, "axes.spines.top": False,
      "axes.spines.right": False, "axes.edgecolor": "#444444", "xtick.color": "#333333",
      "ytick.color": "#333333", "legend.frameon": False, "hatch.color": "white",
      "hatch.linewidth": 1.2, "figure.facecolor": "white", "savefig.facecolor": "white"}  # fmt: skip


def _metric_label(m: str) -> str:
    if m.startswith("response_emotion_"):
        return "Emotion: " + m.removeprefix("response_emotion_")
    return METRIC_LABEL.get(m, m)


def _save(fig, path: Path) -> str:
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _cond_axis(ax, conds, ok: pd.DataFrame | None = None) -> None:
    labels = [
        COND[c]["label"] + ("" if ok is None else f"\nn={(ok['condition'] == c).sum()}")
        for c in conds
    ]
    ax.set_xticks([COND_X[c] for c in conds], labels, fontsize=7.5)
    ax.set_xlim(-0.65, 4.65)


def fig_detection(det: pd.DataFrame, out: Path) -> str | None:
    """Strip plot of -log10 p per generator, on the saved token ids (the retokenized-text
    scores are in the report table)."""
    if det.empty:
        return None
    det = det[det["input_kind"] == "token_ids"] if "token_ids" in set(det["input_kind"]) else det
    gens = sorted(det["generator"].unique()) if "generator" in det else ["all"]
    rng = np.random.default_rng(0)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(gens), figsize=(3.2 * len(gens), 3.2), sharey=True,
                                 squeeze=False, layout="constrained")  # fmt: skip
        for ax, gen in zip(axes[0], gens, strict=True):
            g = det if gen == "all" else det[det["generator"] == gen]
            thr = g["threshold"].iloc[0]
            for x, wm in ((0, False), (1, True)):
                d = g[g["watermark_enabled"] == wm]
                v = -np.log10(d["p_value"].clip(lower=1e-300).to_numpy())
                color = WM if wm else NO_WM
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=16, marker="^" if wm else "o",
                           facecolors=color if wm else "white", edgecolors=color, linewidths=1)  # fmt: skip
            tpr = int((g[g["watermark_enabled"]]["p_value"] < thr).sum())
            fpr = int((g[~g["watermark_enabled"]]["p_value"] < thr).sum())
            n_wm, n_no = int(g["watermark_enabled"].sum()), int((~g["watermark_enabled"]).sum())
            ax.axhline(-np.log10(thr), color="#555555", ls="--", lw=0.9)
            ax.text(-0.45, -np.log10(thr), f"threshold p = {thr:g}", va="bottom", ha="left", fontsize=7, color="#555555")  # fmt: skip
            ax.set_xticks([0, 1], [f"no watermark\n(n={n_no})", f"watermarked\n(n={n_wm})"])
            ax.set_xlim(-0.5, 1.5)
            ax.set_title(f"{gen}\ndetected {tpr}/{n_wm} · false alarms {fpr}/{n_no}")
        axes[0][0].set_ylabel("Evidence of watermark, −log10(p)")
        return _save(fig, out)


def _by_condition(df: pd.DataFrame, metric: str):
    ok = df[df["status"] == "ok"].copy()
    if metric not in ok:
        return ok.iloc[0:0], []
    ok["_v"] = _numeric(ok[metric])
    ok = ok.dropna(subset=["_v"])
    return ok, [c for c in COND_ORDER if c in set(ok["condition"])]


def _bar_by_condition(
    df: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path
) -> str | None:
    """Mean per condition (bar from zero), 95% cluster-bootstrap CI, and one dot per source
    document so the clustering is visible."""
    ok, conds = _by_condition(df, metric)
    if ok.empty:
        return None
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
        for c in conds:
            g, x, st = ok[ok["condition"] == c], COND_X[c], COND[c]
            m = g["_v"].mean()
            ax.bar(x, m, width=0.8, color=st["color"], hatch="///" if st["told"] else None,
                   edgecolor=st["color"], linewidth=0.8)  # fmt: skip
            per_src = g.groupby("source_id")["_v"].mean()
            ax.scatter(np.full(len(per_src), x) + np.linspace(-0.2, 0.2, len(per_src)), per_src,
                       s=12, color="white", edgecolors="#222222", linewidths=0.8, zorder=3)  # fmt: skip
            lo, hi = cluster_bootstrap_ci(g["_v"], g["source_id"])
            if np.isfinite(lo):
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="#222222", capsize=3, lw=1,
                            zorder=4)  # fmt: skip
        _cond_axis(ax, conds, ok)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        n_src = ok["source_id"].nunique()
        fig.supxlabel(f"Bars: mean. Whiskers: 95% bootstrap CI over {n_src} source articles. "
                      "Dots: per-article means.", fontsize=6.5, color="#555555")  # fmt: skip
        return _save(fig, out)


def fig_lengths(df: pd.DataFrame, out: Path) -> str | None:
    ok, conds = _by_condition(df, "metrics.response_words")
    if ok.empty:
        return None
    rng = np.random.default_rng(1)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
        for c in conds:
            d, x, st = ok[ok["condition"] == c]["_v"].to_numpy(), COND_X[c], COND[c]
            ax.scatter(x + rng.uniform(-0.28, 0.28, len(d)), d, s=5, alpha=0.35, linewidths=0,
                       color=st["color"])  # fmt: skip
            bp = ax.boxplot(d, positions=[x], widths=0.65, patch_artist=True, showfliers=False,
                            medianprops={"color": "#111111", "lw": 1.4})  # fmt: skip
            bp["boxes"][0].set(facecolor="none", edgecolor="#222222",
                               hatch="///" if st["told"] else None)  # fmt: skip
        _cond_axis(ax, conds, ok)
        ax.set_ylabel("Receiver response length (words)")
        ax.set_title("Response length by condition")
        return _save(fig, out)


def fig_diffs(cmp: pd.DataFrame, prefixes: tuple[str, ...], title: str, out: Path) -> str | None:
    """One small panel per metric (own x-scale); rows inside a panel are the four contrasts.
    Filled = watermark contrasts, open = telling-the-receiver contrasts."""
    if cmp.empty:
        return None
    d = cmp[cmp["metric"].str.startswith(prefixes)]
    if d.empty:
        return None
    metrics = list(dict.fromkeys(d["metric"]))
    contrasts = [c for c in CONTRAST_LABEL if c in set(d["contrast"])]
    ncols = 2 if len(metrics) == 4 else min(3, len(metrics))
    nrows = -(-len(metrics) // ncols)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols + 1.6, 1.25 * nrows + 0.9),
                                 sharey=True, squeeze=False, layout="constrained")  # fmt: skip
        for ax in axes.flat[len(metrics) :]:
            ax.set_visible(False)
        for ax, m in zip(axes.flat, metrics, strict=False):
            dm = d[d["metric"] == m].set_index("contrast")
            for y, c in enumerate(contrasts):
                if c not in dm.index:
                    continue
                r = dm.loc[c]
                wm_contrast = c in ("C1-C0", "C3-C2")
                lo, hi = r["ci95_low"], r["ci95_high"]
                if np.isfinite(lo):
                    ax.plot([lo, hi], [y, y], color="#222222", lw=1)
                ax.scatter(r["mean_diff"], y, s=22, zorder=3, color="#222222" if wm_contrast else "white",
                           edgecolors="#222222", linewidths=1)  # fmt: skip
            ax.axvline(0, color="#999999", lw=0.8)
            ax.set_title(_metric_label(m), fontsize=8, loc="left")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, symmetric=True))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2g}"))
            ax.tick_params(axis="x", labelsize=7)
        axes[0][0].set_yticks(range(len(contrasts)), [CONTRAST_LABEL[c] for c in contrasts])
        axes[0][0].set_ylim(len(contrasts) - 0.5, -0.5)
        n_src = int(d["n_sources"].max())
        fig.suptitle(title, fontsize=9.5, fontweight="bold")
        fig.supxlabel(f"Paired mean difference (treated − control), 95% cluster-bootstrap CI over "
                      f"{n_src} source articles", fontsize=7.5)  # fmt: skip
        return _save(fig, out)


def fig_model_condition(
    df: pd.DataFrame, out: Path, metric: str = "metrics.task_success"
) -> str | None:
    """Dot plot: one row per generator → receiver pair, one marker per condition."""
    ok, conds = _by_condition(df, metric)
    if ok.empty:
        return None
    tab = ok.pivot_table(index=["generator", "receiver"], columns="condition", values="_v",
                         aggfunc="mean")  # fmt: skip
    offs = {"C0": -0.24, "C1": -0.08, "C2": 0.08, "C3": 0.24}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.4, 0.55 * len(tab) + 1.9), layout="constrained")
        for c in conds:
            st = COND[c]
            ax.scatter(tab[c], np.arange(len(tab)) + offs[c], s=30, marker="s" if st["told"] else "o",
                       facecolors="white" if st["told"] else st["color"], edgecolors=st["color"],
                       linewidths=1.3, label=st["label"].replace("\n", " · ", 1).replace("\n", ", "),
                       zorder=3)  # fmt: skip
        ax.set_yticks(range(len(tab)), [f"{g} → {r}" for g, r in tab.index])
        ax.set_ylim(len(tab) - 0.5, -0.5)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_xlabel("Task success rate (mean over trials)")
        fig.suptitle(
            "Task success by model pair (generator → receiver)", fontsize=9, fontweight="bold"
        )
        fig.legend(loc="outside lower center", ncols=2, fontsize=7)
        return _save(fig, out)


def _cell(v) -> str:
    if isinstance(v, float | np.floating):
        return "NaN" if not np.isfinite(v) else f"{v:.4g}"
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_cell(x) if x is not None else "–" for x in v) + "]"
    return str(v).replace("|", "\\|")


def _fmt_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df.empty:
        return "_none_\n"
    d = df.head(max_rows)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in d.itertuples(index=False)]
    more = [f"\n_{len(df) - max_rows} more rows in the CSV tables_"] if len(df) > max_rows else []
    return "\n".join([head, sep, *body, *more]) + "\n"


def build_report(run_path: str | Path) -> Path:
    run = Path(run_path)
    df = pd.read_parquet(run / "trials.parquet")

    class _RD:  # minimal shim so write_summary can run from a path
        path = run
        run_id = run.name

    summary = write_summary(_RD, df)
    det_path = run / "tables" / "detections.parquet"
    det = pd.read_parquet(det_path) if det_path.exists() else pd.DataFrame()
    cmp = compare(df)
    cmp_by_model = compare(df, group_by=["generator", "receiver"])
    cmp_by_domain = compare(df, group_by=["domain"])
    for name, t in (
        ("comparisons", cmp),
        ("comparisons_by_model", cmp_by_model),
        ("comparisons_by_domain", cmp_by_domain),
    ):
        t.to_csv(run / "tables" / f"{name}.csv", index=False)
        t.to_parquet(run / "tables" / f"{name}.parquet", index=False)
    (run / "metrics" / "comparisons.json").write_text(cmp.to_json(orient="records", indent=2))

    figs = run / "figures"
    made = {
        "detection": fig_detection(det, figs / "detection.png"),
        "contradiction": _bar_by_condition(
            df,
            "metrics.resp_vs_source_contradiction_rate",
            "share of claims contradicted",
            "Response vs source contradiction",
            figs / "contradiction.png",
        ),  # fmt: skip
        "sentiment": fig_diffs(
            cmp,
            ("response_sentiment_negative", "response_emotion"),
            "Sentiment and emotion classifier differences",
            figs / "sentiment.png",
        ),
        "key_effects": fig_diffs(
            cmp,
            (
                "response_words",
                "task_success",
                "resp_vs_source_contradiction_rate",
                "response_sentiment_negative",
            ),
            "Paired effects on receiver behaviour",
            figs / "key_effects.png",
        ),  # fmt: skip
        "length": fig_lengths(df, figs / "length.png"),
        "task_success": _bar_by_condition(
            df, "metrics.task_success", "success rate", "Task success", figs / "task_success.png"
        ),  # fmt: skip
        "model_condition": fig_model_condition(df, figs / "model_condition.png"),
    }

    events = read_events(run / "events.jsonl")
    failed_events = [e for e in events if e["event"].endswith("failed")]
    errors = [
        (p.stem, k, v)
        for p in sorted((run / "artifacts" / "evaluations").glob("*.json"))
        for k, v in json.loads(p.read_text()).get("errors", {}).items()
    ]
    key = ["response_words", "task_success", "marker_present", "resp_vs_source_contradiction_rate",
           "response_sentiment_negative", "response_emotion_fear", "doc_vs_source_similarity"]  # fmt: skip
    cm = (
        pd.DataFrame(summary["condition_means"]).T if summary["condition_means"] else pd.DataFrame()
    )
    cm = (
        cm[[c for c in key if c in cm.columns]].reset_index(names="condition")
        if not cm.empty
        else cm
    )

    lines = [
        f"# Report for run `{run.name}`\n",
        "Generated by `wm-eval report` from saved artifacts only.\n",
        "## Completeness\n",
        f"- Trials planned: {summary['n_trials_planned']}",
        f"- Status counts: {summary['status_counts']}",
        f"- Failed attempts (all stages, retried on resume): {summary['failed_attempts_total']}",
        f"- Failure events in events.jsonl: {len(failed_events)}",
        f"- Evaluator errors: {len(errors)}"
        + "".join(f"\n  - {a} / {b}: {c}" for a, b, c in errors[:20]),
        "",
        "## Watermark detection\n",
        "p-value = P(score ≥ observed | not watermarked with this key). It is not P(watermarked).\n",
    ]
    if len(det):
        groups = [("all", det)]
        if "generator" in det and det["generator"].nunique() > 1:
            groups += list(det.groupby("generator"))
        rates = {(gname, k): v for gname, g in groups for k, v in detection_rates(g).items()}
        rows = [{"generator": k[0], "input": k[1], "threshold": v["threshold"],
                 "TPR": v["tpr"]["rate"], "TPR n": v["tpr"]["n"], "TPR 95% CI": v["tpr"]["ci95"],
                 "FPR": v["fpr"]["rate"], "FPR n": v["fpr"]["n"], "FPR 95% CI": v["fpr"]["ci95"],
                 "median p (WM)": v["median_p_watermarked"], "median p (no WM)": v["median_p_unwatermarked"]}
                for k, v in rates.items()]  # fmt: skip
        lines.append(_fmt_table(pd.DataFrame(rows)))
        lines.append(_fmt_table(det[["doc_id", "input_kind", "watermark_enabled", "p_value", "watermark_detected",
                                     "mean_g", "n_scored"]], max_rows=24))  # fmt: skip
    else:
        lines.append("_detector not run_\n")
    lines += [
        "## Condition means (successful trials)\n",
        _fmt_table(cm),
        "## Paired contrasts\n",
        "Differences are treated − control within pairs that share source, generator, task, receiver "
        "and receiver seed. CIs resample source documents (cluster bootstrap); `p_signflip` is a sign-flip "
        "test on per-source mean differences. With fewer than 2 sources the CI is undefined (NaN) and "
        "p-values are uninformative. These are descriptive contrasts for the configured conditions only.\n",
        _fmt_table(
            cmp[
                [
                    "contrast",
                    "metric",
                    "n_pairs",
                    "n_sources",
                    "mean_diff",
                    "median_diff",
                    "ci95_low",
                    "ci95_high",
                    "p_signflip",
                ]
            ]
            if len(cmp)
            else cmp
        ),  # fmt: skip
        "Grouped tables: `tables/comparisons_by_model.csv`, `tables/comparisons_by_domain.csv`.\n",
        "## Figures\n",
        *[f"![{k}](figures/{v})" for k, v in made.items() if v],
        "",
        "## Caveats\n",
        "- Sentiment, emotion and NLI scores are classifier outputs, not ground truth; validate against "
        "manual annotations before interpreting them.",
        "- Watermarked and unwatermarked documents share a seed but are different samples; differences in "
        "receiver behaviour can come from content differences, not only from the watermark signal.",
        "- llama.cpp sampling with a fixed seed is reproducible only on the same build, model file and "
        "hardware; GPU kernels can be nondeterministic.",
    ]
    report = run / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report
