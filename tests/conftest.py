"""Shared fixtures. Everything is written to pytest's tmp dirs: tests never touch data/ or models/."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("income")
    (root / "data").mkdir()
    for f in ("transaction_types.csv", "mcc_group_definition.csv"):  # small, read-only copies
        shutil.copy(REPO_DATA / f, root / "data" / f)
    return root


@pytest.fixture(scope="session", autouse=True)
def isolated_paths(workdir):
    mp = pytest.MonkeyPatch()
    for var, sub in [
        ("INCOME_DATA_DIR", "data"),
        ("INCOME_MODELS_DIR", "models"),
        ("INCOME_RESULTS_DIR", "results"),
        ("INCOME_IMG_DIR", "img"),
    ]:
        mp.setenv(var, str(workdir / sub))
    mp.delenv("INCOME_MODEL_PATH", raising=False)
    yield
    mp.undo()


def make_transactions(n_users: int = 60, months=("2016-02", "2016-03", "2016-04", "2016-05"), seed: int = 0):
    """Synthetic but structured: each user has a salary level and a spending level."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        salary = rng.choice([0, 800, 2500])
        spend_rate = rng.integers(3, 25)
        for m in months:
            start = pd.Period(m, "M").start_time
            if salary:
                rows.append((f"u{u:03d}", start + pd.Timedelta(days=int(rng.integers(0, 5))), "CT", np.nan, salary))
            for _ in range(rng.poisson(spend_rate)):
                rows.append(
                    (
                        f"u{u:03d}",
                        start + pd.Timedelta(days=int(rng.integers(0, 28))),
                        rng.choice(["PT", "DD", "DT"]),
                        rng.choice([1, 2, 3, 4, np.nan]),
                        int(rng.integers(3, 200)),
                    )
                )
    df = pd.DataFrame(
        rows, columns=["user_id", "transaction_date", "transaction_type", "mcc_group", "amount_n26_currency"]
    )
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@pytest.fixture(scope="session")
def transactions() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture(scope="session")
def lookups(workdir):
    from src.preprocessing import load_lookups

    return load_lookups(workdir / "data")


@pytest.fixture(scope="session")
def trained(transactions, lookups):
    from src.train_model import TrainConfig, run_training

    cfg = TrainConfig(n_clusters=3, n_trials=0, min_cluster_rows=20, holdout_frac=0.25, n_jobs=1)
    return run_training(transactions, lookups, cfg)


@pytest.fixture(scope="session")
def bundle_path(trained, workdir) -> Path:
    from src.bundle import save_bundle

    return save_bundle(trained[0], workdir / "models" / "model_bundle.joblib")
