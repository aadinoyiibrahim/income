"""Training: per-cluster XGBoost regressors for next-month income and expenses.

``run_training`` is the end-to-end, leakage-free protocol:

1. Split *users* into train / hold-out (default 80/20, seeded).
2. **Evaluation fit** - fit clusterer + models on train users using only months
   before the final month M (targets run up to M-1).
3. **Evaluate** - for hold-out users, build features from months < M,
   forecast M, compare with the actual M, and with naive baselines.
4. **Final fit** - refit on *all* users and *all* months with the
   hyper-parameters found in step 2, so production forecasts M+1.

Fixes vs the previous version: Optuna now minimises *cross-validated* RMSE
(it previously minimised in-sample R^2, i.e. chose the worst model); feature
rows are aligned with targets by construction; expenses models are saved for
every cluster; every cluster without enough data uses a global model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from .bundle import GLOBAL, ModelBundle
from .cluster import cluster_users, fit_clusterer, user_profiles
from .config import COL_USER, TARGETS
from .evaluate import add_baselines, bootstrap_mae_diff, metrics_table, per_cluster_table
from .predict import predict_from_features
from .preprocessing import (
    Lookups,
    annotate_transactions,
    build_feature_frame,
    feature_columns,
    monthly_panel,
)

log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
}


@dataclass
class TrainConfig:
    n_clusters: int = 5
    n_trials: int = 25  # Optuna trials per (cluster, target); 0 -> DEFAULT_PARAMS
    cv_folds: int = 3
    holdout_frac: float = 0.2
    min_cluster_rows: int = 200  # smaller clusters use the global model
    seed: int = 42
    n_jobs: int = -1


def make_regressor(params: dict[str, Any], seed: int, n_jobs: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=n_jobs, **params
    )


def tune_params(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: TrainConfig) -> dict[str, Any]:
    """Bayesian (TPE) search minimising user-grouped K-fold RMSE.

    Folds are grouped by user so rows of one user never sit on both sides of a split.
    """
    n_groups = len(np.unique(groups))
    if cfg.n_trials <= 0 or n_groups < cfg.cv_folds or len(y) < 50:
        return dict(DEFAULT_PARAMS)

    import optuna  # training-only dependency

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(GroupKFold(n_splits=cfg.cv_folds).split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        rmses = []
        for tr, va in folds:
            m = make_regressor(params, cfg.seed, cfg.n_jobs).fit(X[tr], y[tr])
            rmses.append(float(np.sqrt(np.mean((m.predict(X[va]) - y[va]) ** 2))))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=cfg.n_trials)
    return study.best_params


def training_rows(tx_raw: pd.DataFrame, lookups: Lookups, clusterer) -> pd.DataFrame:
    feats = build_feature_frame(tx_raw, lookups, with_targets=True)
    feats = cluster_users(feats, clusterer)
    return feats.dropna(subset=[f"target_{t}" for t in TARGETS])


def fit_bundle(
    tx_raw: pd.DataFrame,
    lookups: Lookups,
    cfg: TrainConfig,
    params: dict[str, dict] | None = None,
    clusterer=None,
) -> ModelBundle:
    """Fit clusterer (unless given) + per-cluster and global regressors.

    Hyper-parameters are tuned unless ``params`` is given. ``params`` is keyed by
    cluster id, so pass the matching ``clusterer`` too, or ids lose their meaning.
    """
    if clusterer is None:
        tx = annotate_transactions(tx_raw, lookups)
        profiles = user_profiles(monthly_panel(tx, lookups.categories))
        clusterer = fit_clusterer(profiles, cfg.n_clusters, cfg.seed)
    rows = training_rows(tx_raw, lookups, clusterer)
    fcols = feature_columns(lookups.categories)
    log.info("Training rows: %d user-months from %d users", len(rows), rows[COL_USER].nunique())

    models: dict[str, dict] = {t: {} for t in TARGETS}
    used_params: dict[str, dict] = {t: {} for t in TARGETS}
    groups_all = [(GLOBAL, rows)] + [(int(c), g) for c, g in rows.groupby("cluster") if len(g) >= cfg.min_cluster_rows]
    for target in TARGETS:
        for key, g in groups_all:
            X = g[fcols].to_numpy(dtype=float)
            y = g[f"target_{target}"].to_numpy(dtype=float)
            p = (params or {}).get(target, {}).get(key) or tune_params(X, y, g[COL_USER].to_numpy(), cfg)
            models[target][key] = make_regressor(p, cfg.seed, cfg.n_jobs).fit(X, y)
            used_params[target][key] = p
            log.info("Fitted %-8s model for cluster %-6s (%d rows)", target, key, len(g))

    return ModelBundle(
        lookups=lookups,
        feature_columns=fcols,
        clusterer=clusterer,
        models=models,
        params=used_params,
        trained_through=str(pd.to_datetime(tx_raw["transaction_date"]).max().to_period("M")),
    )


def split_users(users: np.ndarray, holdout_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    users = np.array(sorted(users))
    holdout = rng.choice(users, size=int(round(len(users) * holdout_frac)), replace=False)
    return np.setdiff1d(users, holdout), holdout


def evaluate_holdout(bundle: ModelBundle, tx_holdout: pd.DataFrame, target_period: pd.Period) -> pd.DataFrame:
    """Forecast ``target_period`` for hold-out users from strictly earlier months."""
    dates = pd.to_datetime(tx_holdout["transaction_date"])
    past = tx_holdout[dates.dt.to_period("M") < target_period]
    feats = build_feature_frame(past, bundle.lookups, end_period=target_period - 1)
    preds = predict_from_features(bundle, feats)
    keep = [COL_USER, *TARGETS, "income_roll3_mean", "expenses_roll3_mean"]
    last = feats.groupby(COL_USER).tail(1)[keep]
    frame = preds.merge(last, on=COL_USER)

    full = build_feature_frame(tx_holdout, bundle.lookups, end_period=target_period)
    actual = full.loc[full["period"] == target_period, [COL_USER, *TARGETS]]
    actual = actual.rename(columns={t: f"actual_{t}" for t in TARGETS})
    frame = frame.merge(actual, on=COL_USER, how="inner")
    frame["forecast_month"] = str(target_period)
    return add_baselines(frame)


def run_training(tx_raw: pd.DataFrame, lookups: Lookups, cfg: TrainConfig) -> tuple[ModelBundle, pd.DataFrame]:
    """Evaluate on hold-out users, then refit on everything. Returns (final bundle, hold-out frame)."""
    periods = pd.to_datetime(tx_raw["transaction_date"]).dt.to_period("M")
    last_period = periods.max()
    if periods.nunique() < 3:
        raise ValueError("Need at least 3 months of data (2 to train on, 1 to evaluate).")

    train_users, holdout_users = split_users(tx_raw[COL_USER].unique(), cfg.holdout_frac, cfg.seed)
    log.info("Users: %d train / %d hold-out; evaluating on %s", len(train_users), len(holdout_users), last_period)

    is_train = tx_raw[COL_USER].isin(train_users)
    eval_bundle = fit_bundle(tx_raw[is_train & (periods < last_period)], lookups, cfg)
    holdout = evaluate_holdout(eval_bundle, tx_raw[~is_train], last_period)
    metrics = metrics_table(holdout)
    log.info("Hold-out metrics (%s):\n%s", last_period, metrics.to_string(index=False))

    # Reuse the evaluated clusterer so cluster ids (and their tuned params) keep
    # their meaning, and production clusters match what was evaluated.
    log.info("Refitting on all users/months with tuned hyper-parameters ...")
    final = fit_bundle(tx_raw, lookups, cfg, params=eval_bundle.params, clusterer=eval_bundle.clusterer)
    final.metrics = {
        "evaluated_month": str(last_period),
        "n_holdout_users": int(holdout[COL_USER].nunique()),
        "overall": metrics.to_dict(orient="records"),
        "per_cluster": per_cluster_table(holdout).to_dict(orient="records"),
        "mae_diff_vs_mean_3m": [bootstrap_mae_diff(holdout, t) for t in TARGETS],
    }
    return final, holdout
