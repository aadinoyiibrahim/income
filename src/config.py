"""Central configuration: filesystem locations and pipeline constants.

Every path can be overridden with an environment variable so the same code
runs from a laptop, inside Docker, or in CI without editing source files.

| Variable              | Default           |
|-----------------------|-------------------|
| ``INCOME_DATA_DIR``   | ``<repo>/data``   |
| ``INCOME_MODELS_DIR`` | ``<repo>/models`` |
| ``INCOME_RESULTS_DIR``| ``<repo>/results``|
| ``INCOME_IMG_DIR``    | ``<repo>/img``    |
| ``INCOME_MODEL_PATH`` | ``<models>/model_bundle.joblib`` |
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw-data column contract (N26 export schema).
COL_USER = "user_id"
COL_DATE = "transaction_date"
COL_TYPE = "transaction_type"
COL_MCC = "mcc_group"
COL_AMOUNT = "amount_n26_currency"
REQUIRED_COLUMNS = (COL_USER, COL_DATE, COL_TYPE, COL_AMOUNT)

TARGETS = ("income", "expenses")
UNKNOWN_CATEGORY = "Unknown"
DEFAULT_TRAINING_FILE = "2016-09-19_79351_training.csv"


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    models_dir: Path
    results_dir: Path
    img_dir: Path
    model_path: Path

    @property
    def training_csv(self) -> Path:
        return self.data_dir / DEFAULT_TRAINING_FILE

    @property
    def transaction_types_csv(self) -> Path:
        return self.data_dir / "transaction_types.csv"

    @property
    def mcc_groups_csv(self) -> Path:
        return self.data_dir / "mcc_group_definition.csv"


def get_paths() -> Paths:
    """Resolve paths at call time (not import time) so env overrides in tests work."""
    data_dir = _env_path("INCOME_DATA_DIR", PROJECT_ROOT / "data")
    models_dir = _env_path("INCOME_MODELS_DIR", PROJECT_ROOT / "models")
    return Paths(
        data_dir=data_dir,
        models_dir=models_dir,
        results_dir=_env_path("INCOME_RESULTS_DIR", PROJECT_ROOT / "results"),
        img_dir=_env_path("INCOME_IMG_DIR", PROJECT_ROOT / "img"),
        model_path=_env_path("INCOME_MODEL_PATH", models_dir / "model_bundle.joblib"),
    )
