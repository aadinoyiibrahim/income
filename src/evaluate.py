"""Hold-out evaluation against naive baselines.

A forecast model is only useful if it beats trivial rules, so every metric is
reported next to two baselines computed from the same features:

* ``last_month``  - next month equals this month (persistence forecast)
* ``mean_3m``     - next month equals the mean of the last <=3 months
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import TARGETS

BASELINES = {"last_month": "{t}", "mean_3m": "{t}_roll3_mean"}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n": int(len(y_true)),
    }


def add_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for t in TARGETS:
        for name, col in BASELINES.items():
            out[f"{name}_{t}"] = out[col.format(t=t)]
    return out


def metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Long table: target x method x (MAE, RMSE, R2, n). ``frame`` needs actual_*, predicted_*, baselines."""
    rows = []
    for t in TARGETS:
        y = frame[f"actual_{t}"]
        methods = {"model": frame[f"predicted_{t}"], **{b: frame[f"{b}_{t}"] for b in BASELINES}}
        for method, yhat in methods.items():
            rows.append({"target": t, "method": method, **regression_metrics(y, yhat)})
    return pd.DataFrame(rows)


def per_cluster_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, g in frame.groupby("cluster"):
        for t in TARGETS:
            m = regression_metrics(g[f"actual_{t}"], g[f"predicted_{t}"])
            b = regression_metrics(g[f"actual_{t}"], g[f"mean_3m_{t}"])
            rows.append({"cluster": int(cluster), "target": t, **m, "MAE_mean_3m": b["MAE"]})
    return pd.DataFrame(rows)


def bootstrap_mae_diff(
    frame: pd.DataFrame, target: str, baseline: str = "mean_3m", n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap of MAE(model) - MAE(baseline) over hold-out users.

    Negative = model better. If the 95% interval contains 0, the two are not
    distinguishable on this hold-out set.
    """
    y = frame[f"actual_{target}"].to_numpy(float)
    d = np.abs(frame[f"predicted_{target}"].to_numpy(float) - y) - np.abs(
        frame[f"{baseline}_{target}"].to_numpy(float) - y
    )
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "target": target,
        "baseline": baseline,
        "mae_diff": float(d.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }
