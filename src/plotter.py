"""Evaluation and model-inspection figures (requires the ``analysis`` extra).

Every function takes ``outdir_`` (a directory path ending with ``/``) and saves
with ``plt.savefig(outdir_ + '<name>.png', dpi=300, bbox_inches='tight')``.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: CLI, CI and Docker have no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .bundle import GLOBAL, ModelBundle  # noqa: E402
from .config import TARGETS  # noqa: E402
from .evaluate import metrics_table, per_cluster_table  # noqa: E402

# Fixed categorical order (colour follows the method, never its rank).
COLORS = {"model": "#2a78d6", "last_month": "#eb6834", "mean_3m": "#1baf7a"}
LABELS = {"model": "XGBoost (per cluster)", "last_month": "Baseline: last month", "mean_3m": "Baseline: 3-month mean"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _out(outdir_: str) -> str:
    os.makedirs(outdir_, exist_ok=True)
    return outdir_ if outdir_.endswith("/") else outdir_ + "/"


def plot_evaluation(holdout: pd.DataFrame, outdir_: str) -> str:
    """6-panel hold-out summary: actual vs predicted, MAE vs baselines, per-cluster MAE, residuals."""
    outdir_ = _out(outdir_)
    month = holdout.get("forecast_month", pd.Series(["hold-out month"])).iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    # (a, b) actual vs predicted, log-log because both flows span 3 orders of magnitude
    for ax, t in zip(axes[0, :2], TARGETS):
        a, p = holdout[f"actual_{t}"].to_numpy(), holdout[f"predicted_{t}"].to_numpy()
        ax.scatter(a + 1, p + 1, s=9, alpha=0.35, color=COLORS["model"], edgecolors="none")
        lim = [1, max(a.max(), p.max()) * 1.5]
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="perfect forecast")
        ax.set(
            xscale="log",
            yscale="log",
            xlim=lim,
            ylim=lim,
            xlabel=f"actual {t} + 1 (€)",
            ylabel=f"predicted {t} + 1 (€)",
        )
        ax.set_title(f"({'ab'[TARGETS.index(t)]}) {t.capitalize()}: actual vs predicted", loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper left")
        _style(ax)

    # (c) MAE: model vs baselines
    ax = axes[0, 2]
    m = metrics_table(holdout)
    methods = list(COLORS)
    x = np.arange(len(TARGETS))
    w = 0.26
    for i, meth in enumerate(methods):
        vals = [m.query("target == @t and method == @meth")["MAE"].item() for t in TARGETS]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=COLORS[meth], label=LABELS[meth])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK, padding=2)
    ax.set_xticks(x, [t.capitalize() for t in TARGETS])
    ax.set(ylabel="MAE (€ per user, lower is better)")
    ax.set_title("(c) Hold-out MAE vs naive baselines", loc="left", fontsize=11)
    ax.set_ylim(0, m["MAE"].max() * 1.45)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    _style(ax)

    # (d, e) per-cluster MAE, model vs the strongest baseline
    pc = per_cluster_table(holdout)
    counts = holdout.groupby("cluster").size()
    for ax, t, tag in zip(axes[1, :2], TARGETS, "de"):
        d = pc[pc["target"] == t].sort_values("cluster")
        xc = np.arange(len(d))
        ax.bar(xc - 0.2, d["MAE"], 0.38, color=COLORS["model"], label=LABELS["model"])
        ax.bar(xc + 0.2, d["MAE_mean_3m"], 0.38, color=COLORS["mean_3m"], label=LABELS["mean_3m"])
        ax.set_xticks(xc, [f"C{c}\nn={counts[c]}" for c in d["cluster"]])
        ax.set(ylabel="MAE (€)")
        ax.set_title(f"({tag}) {t.capitalize()} MAE by user cluster", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)

    # (f) residual distribution (signed): centred near 0 = unbiased
    ax = axes[1, 2]
    for t, color in zip(TARGETS, ("#2a78d6", "#eb6834")):
        r = holdout[f"predicted_{t}"] - holdout[f"actual_{t}"]
        ax.hist(r.clip(-1000, 1000), bins=60, alpha=0.55, color=color, label=f"{t} (median {r.median():+.0f} €)")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set(xlabel="predicted − actual (€, clipped at ±1000)", ylabel="users")
    ax.set_title("(f) Forecast error distribution", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    fig.suptitle(f"Hold-out evaluation — forecasting {month} for unseen users", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    plt.savefig(outdir_ + "evaluation_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outdir_ + "evaluation_overview.png"


def plot_feature_importance(bundle: ModelBundle, outdir_: str, top: int = 15) -> str:
    """Gain importance of the global income and expenses models, side by side."""
    outdir_ = _out(outdir_)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, t, color in zip(axes, TARGETS, ("#2a78d6", "#eb6834")):
        booster = bundle.models[t][GLOBAL].get_booster()
        gain = booster.get_score(importance_type="gain")
        imp = pd.Series({bundle.feature_columns[int(k[1:])]: v for k, v in gain.items()})
        imp = (imp / imp.sum()).sort_values().tail(top)
        ax.barh(imp.index, imp.values, color=color, height=0.7)
        ax.set(xlabel="share of total gain")
        ax.set_title(f"{t.capitalize()} model (global): top {top} features", loc="left", fontsize=11)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    plt.savefig(outdir_ + "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outdir_ + "feature_importance.png"


def plot_shap_summary(bundle: ModelBundle, features: pd.DataFrame, outdir_: str, n: int = 1000) -> str:
    """SHAP beeswarm for the global expenses model on a sample of user-months."""
    import shap

    outdir_ = _out(outdir_)
    X = features[bundle.feature_columns].sample(min(n, len(features)), random_state=0)
    model = bundle.models["expenses"][GLOBAL]
    values = shap.TreeExplainer(model).shap_values(X)
    shap.summary_plot(values, X, max_display=15, show=False)
    plt.title("SHAP: what drives next-month expenses (global model)", loc="left", fontsize=11)
    plt.savefig(outdir_ + "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    return outdir_ + "shap_summary.png"


def plot_clusters(features: pd.DataFrame, outdir_: str) -> str:
    """Cluster profiles: size, and median monthly income / expenses per cluster."""
    from .cluster import user_profiles

    outdir_ = _out(outdir_)
    prof = user_profiles(features).join(features.groupby("user_id")["cluster"].first())
    g = prof.groupby("cluster")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (g.size(), "users", "(a) Users per cluster"),
        (g["mean_income"].median(), "€ / month", "(b) Median monthly income"),
        (g["mean_expenses"].median(), "€ / month", "(c) Median monthly expenses"),
    ]
    for ax, (s, ylabel, title) in zip(axes, panels):
        bars = ax.bar([f"C{c}" for c in s.index], s.values, color="#2a78d6", width=0.6)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        ax.set(ylabel=ylabel)
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    fig.tight_layout()
    plt.savefig(outdir_ + "clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outdir_ + "clusters.png"
