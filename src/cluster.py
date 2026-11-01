"""User segmentation with K-means on per-user monthly spending profiles.

Profiles are *per-month rates* (mean monthly income, expenses, transaction
count, share of incoming transactions), so a user with 2 months of history and
one with 6 months are comparable. Money/count features are log1p-transformed
(heavy right tails) and standardised. The scaler is fitted once and stored with
K-means in a single sklearn ``Pipeline``; the original code re-fitted the
scaler on every inference batch, so cluster ids depended on who else was in
the upload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import COL_USER

PROFILE_COLUMNS = ["mean_income", "mean_expenses", "mean_tx_count", "income_ratio"]


def user_profiles(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per user from a user-month panel (see ``preprocessing.monthly_panel``)."""
    prof = panel.groupby(COL_USER).agg(
        mean_income=("income", "mean"),
        mean_expenses=("expenses", "mean"),
        mean_tx_count=("tx_count", "mean"),
        income_ratio=("income_ratio", "mean"),
    )
    return prof[PROFILE_COLUMNS]


def _log1p_money(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    X[:, :3] = np.log1p(np.clip(X[:, :3], 0, None))
    return X


def fit_clusterer(profiles: pd.DataFrame, n_clusters: int = 5, seed: int = 42) -> Pipeline:
    n_clusters = max(1, min(n_clusters, len(profiles)))
    pipe = make_pipeline(
        FunctionTransformer(_log1p_money),
        StandardScaler(),
        KMeans(n_clusters=n_clusters, n_init=10, random_state=seed),
    )
    pipe.fit(profiles[PROFILE_COLUMNS].to_numpy())
    return pipe


def assign_clusters(clusterer: Pipeline, profiles: pd.DataFrame) -> pd.Series:
    labels = clusterer.predict(profiles[PROFILE_COLUMNS].to_numpy())
    return pd.Series(labels.astype(int), index=profiles.index, name="cluster")


def cluster_users(panel: pd.DataFrame, clusterer: Pipeline) -> pd.DataFrame:
    """Attach a ``cluster`` column to every row of a user-month frame."""
    labels = assign_clusters(clusterer, user_profiles(panel))
    return panel.drop(columns="cluster", errors="ignore").merge(labels, left_on=COL_USER, right_index=True, how="left")
