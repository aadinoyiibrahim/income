"""A single, versioned model artifact: everything needed to serve predictions.

Bundling the lookups, feature list, clusterer and regressors into one file
removes the old failure mode where ``scaler.pkl`` / ``feature_columns.pkl`` /
per-cluster pickles drifted out of sync.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn
import xgboost

from . import __version__
from .preprocessing import Lookups

log = logging.getLogger(__name__)

GLOBAL = "global"


@dataclass
class ModelBundle:
    lookups: Lookups
    feature_columns: list[str]
    clusterer: Any
    # models[target][cluster_id | "global"] -> fitted XGBRegressor
    models: dict[str, dict[int | str, Any]]
    params: dict[str, dict[int | str, dict]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    trained_through: str = ""  # last calendar month of training data, e.g. "2016-07"
    package_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    library_versions: dict[str, str] = field(
        default_factory=lambda: {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
        }
    )

    def model_for(self, target: str, cluster: int) -> tuple[Any, int | str]:
        per_target = self.models[target]
        key: int | str = int(cluster) if int(cluster) in per_target else GLOBAL
        return per_target[key], key

    def summary(self) -> dict[str, Any]:
        return {
            "package_version": self.package_version,
            "created_at": self.created_at,
            "trained_through": self.trained_through,
            "n_features": len(self.feature_columns),
            "clusters_with_own_model": {t: sorted(k for k in m if k != GLOBAL) for t, m in self.models.items()},
            "library_versions": self.library_versions,
            "metrics": self.metrics,
        }


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)  # atomic on POSIX: a reader never sees a half-written file
    log.info("Saved model bundle to %s", path)
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}. Run `income train` first.")
    bundle: ModelBundle = joblib.load(path)
    for lib, current in (("xgboost", xgboost.__version__), ("scikit-learn", sklearn.__version__)):
        trained = bundle.library_versions.get(lib)
        if trained and trained != current:
            log.warning("Bundle trained with %s %s, running %s; re-train if results look off.", lib, trained, current)
    return bundle
