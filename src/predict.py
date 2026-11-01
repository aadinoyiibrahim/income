"""Next-month income / expenses forecasts with category breakdowns."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .bundle import ModelBundle
from .cluster import cluster_users
from .config import COL_DATE, COL_USER, TARGETS
from .preprocessing import build_feature_frame

log = logging.getLogger(__name__)

BREAKDOWN_WINDOW = 3  # months of history used for category shares


def _breakdown(history: pd.DataFrame, prefix: str, total: float, categories: list[str]) -> dict[str, float]:
    """Split ``total`` across categories by their share over the last few months.

    Falls back to ``{"Unknown": total}`` when the user has no history for this flow.
    """
    sums = history[[prefix + c for c in categories]].sum()
    denom = float(sums.sum())
    if denom <= 0:
        return {"Unknown": round(total, 2)}
    shares = sums[sums > 0] / denom
    return {k.removeprefix(prefix): round(float(total * v), 2) for k, v in shares.items()}


def predict_from_features(bundle: ModelBundle, feats: pd.DataFrame) -> pd.DataFrame:
    """Predict month t+1 from each user's **last** row in a feature frame."""
    feats = cluster_users(feats, bundle.clusterer)
    last = feats.groupby(COL_USER, sort=False).tail(1).reset_index(drop=True)
    X = last[bundle.feature_columns].to_numpy(dtype=float)
    out = last[[COL_USER, "period", "cluster"]].copy()
    for target in TARGETS:
        pred = np.zeros(len(last))
        used = np.empty(len(last), dtype=object)
        for cluster in out["cluster"].unique():
            mask = (out["cluster"] == cluster).to_numpy()
            model, key = bundle.model_for(target, cluster)
            pred[mask] = model.predict(X[mask])
            used[mask] = str(key)
        out[f"predicted_{target}"] = np.clip(pred, 0.0, None)  # flows are non-negative
        out[f"model_{target}"] = used
    return out


def predict_next_month(bundle: ModelBundle, transactions: pd.DataFrame) -> pd.DataFrame:
    """Raw transactions (one or more months per user) -> forecast for the following month.

    The forecast month is the calendar month after the last month present in
    ``transactions``. That last month should be complete; a partial month
    biases the lag features (and therefore the forecast) downwards.
    """
    last_date = pd.to_datetime(transactions[COL_DATE]).max()
    if not last_date.is_month_end:
        log.warning("Latest transaction is %s (not a month end): last month may be incomplete.", last_date.date())

    feats = build_feature_frame(transactions, bundle.lookups)
    preds = predict_from_features(bundle, feats)

    cats = bundle.lookups.categories
    recent = feats.groupby(COL_USER, sort=False).tail(BREAKDOWN_WINDOW)
    hist = {u: g for u, g in recent.groupby(COL_USER, sort=False)}
    preds["predicted_income_breakdown"] = [
        _breakdown(hist[u], "inc_", t, cats) for u, t in zip(preds[COL_USER], preds["predicted_income"])
    ]
    preds["predicted_expenses_breakdown"] = [
        _breakdown(hist[u], "exp_", t, cats) for u, t in zip(preds[COL_USER], preds["predicted_expenses"])
    ]
    preds["forecast_month"] = (preds["period"] + 1).astype(str)
    cols = [
        COL_USER,
        "forecast_month",
        "cluster",
        "predicted_income",
        "predicted_expenses",
        "predicted_income_breakdown",
        "predicted_expenses_breakdown",
    ]
    return preds[cols].sort_values(COL_USER).reset_index(drop=True)
