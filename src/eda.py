"""Exploratory data analysis: summaries, two-sample tests and a multi-panel figure.

Requires the ``analysis`` extra (matplotlib, seaborn, scipy).
"""

from __future__ import annotations

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import chi2_contingency, ks_2samp  # noqa: E402

from .config import COL_AMOUNT, COL_DATE, COL_TYPE, COL_USER  # noqa: E402
from .preprocessing import Lookups, annotate_transactions  # noqa: E402

log = logging.getLogger(__name__)
BLUE, ORANGE = "#2a78d6", "#eb6834"
FLOW_COLORS = {"In": BLUE, "Out": ORANGE}


def summarize(tx: pd.DataFrame) -> dict:
    """Descriptive statistics and tests on annotated transactions.

    * KS test, In vs Out amounts: are the amount *distributions* different?
    * Chi-square, flow x category: is category informative about direction?
    * Chi-square, type x month: is the transaction mix stable over time?
      (With ~400k rows these tests reject almost anything; read effect sizes
      in the figure, not just p-values.)
    """
    ks = ks_2samp(tx.loc[tx.flow == "In", COL_AMOUNT], tx.loc[tx.flow == "Out", COL_AMOUNT])
    chi_cat = chi2_contingency(pd.crosstab(tx["flow"], tx["category"]))
    chi_month = chi2_contingency(pd.crosstab(tx[COL_TYPE], tx["period"]))
    return {
        "n_transactions": int(len(tx)),
        "n_users": int(tx[COL_USER].nunique()),
        "date_range": [str(tx[COL_DATE].min().date()), str(tx[COL_DATE].max().date())],
        "flow_share": tx["flow"].value_counts(normalize=True).round(3).to_dict(),
        "unknown_category_share": float((tx["category"] == "Unknown").mean().round(3)),
        "ks_in_vs_out": {"stat": float(ks.statistic), "p": float(ks.pvalue)},
        "chi2_flow_category": {"stat": float(chi_cat[0]), "p": float(chi_cat[1])},
        "chi2_type_month": {"stat": float(chi_month[0]), "p": float(chi_month[1])},
    }


def perform_eda(tx_raw: pd.DataFrame, lookups: Lookups, outdir_: str) -> dict:
    """Write ``eda_overview.png`` into ``outdir_`` and return the summary dict."""
    os.makedirs(outdir_, exist_ok=True)
    outdir_ = outdir_ if outdir_.endswith("/") else outdir_ + "/"
    tx = annotate_transactions(tx_raw, lookups)
    summary = summarize(tx)
    for k, v in summary.items():
        log.info("%-24s %s", k, v)

    month_str = tx["period"].astype(str)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    for flow, c in FLOW_COLORS.items():
        ax.hist(tx.loc[tx.flow == flow, COL_AMOUNT], bins=np.logspace(0.4, 3.1, 50), alpha=0.6, color=c, label=flow)
    ax.set(xscale="log", xlabel="amount (€, log scale)", ylabel="transactions")
    ax.set_title("(a) Amount distribution by flow", loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    monthly = tx.groupby([month_str, "flow"])[COL_AMOUNT].sum().unstack() / 1e6
    for flow, c in FLOW_COLORS.items():
        ax.plot(monthly.index, monthly[flow], marker="o", lw=2, color=c, label=flow)
    ax.set(ylabel="total (M€)")
    ax.set_title("(b) Monthly total income (In) vs expenses (Out)", loc="left")
    ax.legend(frameon=False)

    ax = axes[0, 2]
    active = tx.groupby(month_str)[COL_USER].nunique()
    ax.bar(active.index, active.values, color=BLUE, width=0.6)
    ax.set(ylabel="active users")
    ax.set_title("(c) Active users per month", loc="left")

    ax = axes[1, 0]
    out_cat = tx[tx.flow == "Out"].groupby("category")[COL_AMOUNT].sum().sort_values() / 1e6
    ax.barh(out_cat.index, out_cat.values, color=ORANGE, height=0.7)
    ax.set(xlabel="total expenses (M€)")
    ax.set_title("(d) Expenses by MCC category", loc="left")

    ax = axes[1, 1]
    top = tx[tx.flow == "Out"]["category"].value_counts().index[:8]
    sub = tx[(tx.flow == "Out") & tx["category"].isin(top)]
    sns.boxplot(data=sub, y="category", x=COL_AMOUNT, order=top, ax=ax, color="#9ec5f4", fliersize=1, log_scale=True)
    ax.set(xlabel="amount (€, log scale)", ylabel="")
    ax.set_title("(e) Expense amount per transaction, top-8 categories", loc="left")

    ax = axes[1, 2]
    user = tx.groupby(COL_USER).agg(n=("flow", "size"), total=(COL_AMOUNT, "sum"))
    ax.scatter(user["n"], user["total"], s=5, alpha=0.3, color=BLUE, edgecolors="none")
    ax.set(xscale="log", yscale="log", xlabel="transactions per user", ylabel="total volume per user (€)")
    ax.set_title("(f) Activity vs volume per user", loc="left")

    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{summary['n_transactions']:,} transactions · {summary['n_users']:,} users · "
        f"{summary['date_range'][0]} → {summary['date_range'][1]}",
        x=0.01,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout()
    plt.savefig(outdir_ + "eda_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary
