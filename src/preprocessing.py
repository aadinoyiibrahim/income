"""Data loading, validation and user-month feature engineering.

Pipeline: raw transactions -> annotated transactions (flow, category, period)
-> complete user x month panel -> lag / rolling features -> next-month targets.

All steps are vectorised (groupby / pivot); the original row-wise ``apply``
implementation was O(user-months x transactions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from .config import (
    COL_AMOUNT,
    COL_DATE,
    COL_MCC,
    COL_TYPE,
    COL_USER,
    REQUIRED_COLUMNS,
    UNKNOWN_CATEGORY,
    get_paths,
)

log = logging.getLogger(__name__)

BASE_FEATURES = [
    "income",
    "expenses",
    "tx_count",
    "income_ratio",
    "income_lag1",
    "income_lag2",
    "expenses_lag1",
    "expenses_lag2",
    "income_roll3_mean",
    "expenses_roll3_mean",
    "income_roll3_std",
    "expenses_roll3_std",
    "months_active",
]


@dataclass
class Lookups:
    """Mapping tables shipped inside the model bundle (no data/ dir needed at serve time)."""

    type_to_flow: dict[str, str]
    mcc_to_category: dict[int, str]
    categories: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.categories:
            cats = sorted(set(self.mcc_to_category.values()))
            self.categories = [*cats, UNKNOWN_CATEGORY] if UNKNOWN_CATEGORY not in cats else cats


def load_lookups(data_dir: str | Path | None = None) -> Lookups:
    """Read ``transaction_types.csv`` and ``mcc_group_definition.csv``.

    Transaction types whose direction is not ``In``/``Out`` (e.g. authorisation
    rejects, direction ``-``) are excluded: they move no money.
    """
    data_dir = Path(data_dir) if data_dir else get_paths().data_dir
    type_df = pd.read_csv(data_dir / "transaction_types.csv")
    mcc_df = pd.read_csv(data_dir / "mcc_group_definition.csv")
    type_to_flow = {str(t): d for t, d in zip(type_df["type"], type_df["direction"]) if d in ("In", "Out")}
    mcc_to_category = {int(k): str(v) for k, v in zip(mcc_df["mcc_group"], mcc_df["explanation"])}
    return Lookups(type_to_flow=type_to_flow, mcc_to_category=mcc_to_category)


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse ISO (``2016-02-01``) and German short (``01.02.16``) dates.

    The shipped CSVs use both formats; ``format='mixed', dayfirst=True`` silently
    mis-parses ISO dates, so each format is tried explicitly.
    """
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d.%m.%y", "%d.%m.%Y"):
        missing = out.isna()
        if not missing.any():
            break
        out[missing] = pd.to_datetime(s[missing], format=fmt, errors="coerce")
    if out.isna().any():
        bad = s[out.isna()].unique()[:5]
        raise ValueError(f"Unparseable {COL_DATE} values, e.g. {list(bad)}")
    return out


def validate_transactions(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required: {list(REQUIRED_COLUMNS)}")
    if df.empty:
        raise ValueError("No transactions in input.")
    if not pd.api.types.is_numeric_dtype(df[COL_AMOUNT]):
        raise ValueError(f"Column {COL_AMOUNT!r} must be numeric.")


def read_transactions(source: str | Path | IO) -> pd.DataFrame:
    """Read a raw transaction CSV and validate the column contract."""
    df = pd.read_csv(source, dtype={COL_USER: str, COL_TYPE: str})
    validate_transactions(df)
    df[COL_DATE] = parse_dates(df[COL_DATE])
    return df


def annotate_transactions(df: pd.DataFrame, lookups: Lookups) -> pd.DataFrame:
    """Add ``flow`` (In/Out), ``category`` (MCC explanation) and monthly ``period``."""
    validate_transactions(df)
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[COL_DATE]):
        out[COL_DATE] = parse_dates(out[COL_DATE])
    out["flow"] = out[COL_TYPE].map(lookups.type_to_flow)
    n_unmapped = int(out["flow"].isna().sum())
    if n_unmapped:
        log.warning(
            "Dropping %d transactions with non-monetary/unknown types: %s",
            n_unmapped,
            sorted(out.loc[out["flow"].isna(), COL_TYPE].astype(str).unique())[:10],
        )
        out = out[out["flow"].notna()]
    mcc = pd.to_numeric(out[COL_MCC], errors="coerce") if COL_MCC in out else pd.Series(np.nan, index=out.index)
    out["category"] = mcc.map(lambda v: lookups.mcc_to_category.get(int(v)) if pd.notna(v) else None)
    out["category"] = out["category"].fillna(UNKNOWN_CATEGORY)
    out["period"] = out[COL_DATE].dt.to_period("M")
    return out


def monthly_panel(tx: pd.DataFrame, categories: list[str], end_period: pd.Period | None = None) -> pd.DataFrame:
    """Aggregate annotated transactions to a *complete* user x month panel.

    Each user gets one row per month from their first observed month to
    ``end_period`` (default: last month in the data). Months without
    transactions are real zeros, not missing data, so the next-month target is
    well defined even for inactive months.
    """
    end_period = end_period or tx["period"].max()
    amt = tx.groupby([COL_USER, "period", "flow", "category"])[COL_AMOUNT].sum()
    wide = amt.unstack(["flow", "category"], fill_value=0.0)
    cols = {}
    for flow, prefix in (("In", "inc_"), ("Out", "exp_")):
        for cat in categories:
            key = (flow, cat)
            cols[prefix + cat] = wide[key] if key in wide.columns else 0.0
    panel = pd.DataFrame(cols, index=wide.index)
    counts = tx.groupby([COL_USER, "period"])["flow"].agg(tx_count="size", n_in=lambda s: (s == "In").sum())
    panel = panel.join(counts)

    # Complete the grid: first active month .. end_period, for every user.
    first = tx.groupby(COL_USER)["period"].min()
    idx = pd.MultiIndex.from_tuples(
        [(u, p) for u, p0 in first.items() for p in pd.period_range(p0, end_period, freq="M")],
        names=[COL_USER, "period"],
    )
    panel = panel.reindex(idx, fill_value=0).reset_index()
    inc_cols = [c for c in panel if c.startswith("inc_")]
    exp_cols = [c for c in panel if c.startswith("exp_")]
    panel["income"] = panel[inc_cols].sum(axis=1)
    panel["expenses"] = panel[exp_cols].sum(axis=1)
    panel["tx_count"] = panel["tx_count"].astype(int)
    panel["income_ratio"] = np.where(panel["tx_count"] > 0, panel["n_in"] / panel["tx_count"].clip(lower=1), 0.0)
    return panel.sort_values([COL_USER, "period"]).reset_index(drop=True)


def feature_columns(categories: list[str]) -> list[str]:
    return BASE_FEATURES + [f"avg_exp_{c}" for c in categories]


def add_features(panel: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Lag and 3-month rolling features, computed strictly from months <= t."""
    df = panel.sort_values([COL_USER, "period"]).copy()
    g = df.groupby(COL_USER, sort=False)
    for col in ("income", "expenses"):
        df[f"{col}_lag1"] = g[col].shift(1).fillna(0.0)
        df[f"{col}_lag2"] = g[col].shift(2).fillna(0.0)
        roll = g[col].rolling(3, min_periods=1)
        df[f"{col}_roll3_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_roll3_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
    df["months_active"] = g.cumcount() + 1
    exp_cols = [f"exp_{c}" for c in categories]
    rolled = g[exp_cols].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    for c in categories:
        df[f"avg_exp_{c}"] = rolled[f"exp_{c}"]
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """``target_<x>`` = value of ``<x>`` in month t+1 (NaN for each user's last month)."""
    out = df.copy()
    g = out.groupby(COL_USER, sort=False)
    for col in ("income", "expenses"):
        out[f"target_{col}"] = g[col].shift(-1)
    return out


def build_feature_frame(
    tx_raw: pd.DataFrame,
    lookups: Lookups,
    end_period: pd.Period | None = None,
    with_targets: bool = False,
) -> pd.DataFrame:
    """Raw transactions -> user-month feature frame (optionally with targets)."""
    tx = annotate_transactions(tx_raw, lookups)
    if tx.empty:
        raise ValueError("No monetary transactions left after mapping transaction types.")
    panel = monthly_panel(tx, lookups.categories, end_period=end_period)
    feats = add_features(panel, lookups.categories)
    return add_targets(feats) if with_targets else feats
