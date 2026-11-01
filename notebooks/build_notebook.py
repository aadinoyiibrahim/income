"""Regenerate notebooks/analysis.ipynb (source of truth for the notebook's cells).

    uv run python notebooks/build_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Income & expenses forecasting: analysis notebook

This notebook walks through the pipeline in `src/` on the real data: EDA → features → clusters →
training and hold-out evaluation → August forecasts. It calls the package code directly, so it shows
exactly what the CLI and API do.

**Running it efficiently**
* `RETRAIN = False` (default) loads `models/model_bundle.joblib` and `results/holdout_predictions.csv`
  from `uv run income train`. The whole notebook then runs in about 30 s.
* `RETRAIN = True` re-runs the full protocol: about 30 s with `N_TRIALS = 0` (default
  hyper-parameters), about 4 min with `N_TRIALS = 25` (what produced the shipped model).
* Launch with `uv run jupyter lab notebooks/analysis.ipynb` (after `uv sync --all-extras`)."""),
    code("""RETRAIN = False   # True: re-run hold-out evaluation + final fit
N_TRIALS = 25     # Optuna trials per model when RETRAIN (0 = fast defaults)"""),
    code("""import json, logging
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from src import plotter
from src.bundle import load_bundle, save_bundle
from src.cluster import cluster_users
from src.config import get_paths
from src.eda import perform_eda
from src.evaluate import bootstrap_mae_diff, metrics_table, per_cluster_table
from src.predict import predict_next_month
from src.preprocessing import build_feature_frame, load_lookups, read_transactions
from src.train_model import TrainConfig, run_training

%matplotlib inline
%config InlineBackend.figure_format = "retina"
logging.basicConfig(level=logging.WARNING)
pd.set_option("display.float_format", "{:,.1f}".format)
paths = get_paths()
outdir_ = str(paths.img_dir) + "/"

lookups = load_lookups()
tx = read_transactions(paths.training_csv)
print(f"{len(tx):,} transactions, {tx.user_id.nunique():,} users, "
      f"{tx.transaction_date.min().date()} → {tx.transaction_date.max().date()}")"""),
    md("""## 1. Exploratory data analysis

**What to expect:** amounts are bounded (3–1,166 €) and right-skewed. There are more outgoing
than incoming transactions. About 37% of transactions have no MCC (transfers), so `Unknown` is the
largest category. Active users grow each month (accounts opening), which is why the panel starts each
user at their *first* month rather than at February."""),
    code("""summary, fig = perform_eda(tx, lookups, outdir_ + "eda/")  # also saved to img/eda/eda_overview.png
print(json.dumps({k: summary[k] for k in ("flow_share", "unknown_category_share", "ks_in_vs_out")}, indent=1))
plt.show()"""),
    md("""## 2. Load or train the model

`run_training` is leakage-free. It (i) splits **users** 80/20; (ii) fits clusters and models on the
training users using months **before July**; (iii) forecasts **July** for the unseen hold-out users from
their Feb–Jun history; (iv) refits on everyone and every month, so production forecasts **August**."""),
    code("""if RETRAIN:
    bundle, holdout = run_training(tx, lookups, TrainConfig(n_trials=N_TRIALS))
    save_bundle(bundle, paths.model_path)
    paths.results_dir.mkdir(exist_ok=True)
    holdout.to_csv(paths.results_dir / "holdout_predictions.csv", index=False)
else:
    holdout_csv = paths.results_dir / "holdout_predictions.csv"
    if not holdout_csv.exists():
        raise FileNotFoundError(f"{holdout_csv} missing: run `uv run income train` or set RETRAIN = True")
    bundle = load_bundle(paths.model_path)
    holdout = pd.read_csv(holdout_csv)

feats = cluster_users(build_feature_frame(tx, lookups), bundle.clusterer)
print("trained through", bundle.trained_through, "| features:", len(bundle.feature_columns))"""),
    md("""## 3. Toy example: one real user per cluster

Concept: each row is a **user-month**. Features at month *t* use only months ≤ *t* (lags, 3-month rolling
means, per-category spending). The target is the *next* month's total. Below, the user nearest the median
monthly expenses of each cluster.

**What to expect:** `income_lag1` in a row equals the previous row's `income`. `income_roll3_mean`
is the mean of up to three visible months. Clusters separate users mainly by activity level and by
whether a salary-like inflow is present."""),
    code("""prof = feats.groupby("user_id").agg(cluster=("cluster", "first"), med_exp=("expenses", "median"))
toy_users = (prof.assign(d=lambda d: (d.med_exp - d.groupby("cluster").med_exp.transform("median")).abs())
                 .sort_values("d").groupby("cluster").head(1).sort_values("cluster").index.tolist())
cols = ["user_id", "cluster", "period", "income", "expenses", "tx_count", "income_lag1", "income_roll3_mean",
        "expenses_roll3_mean"]
display(feats.loc[feats.user_id.isin(toy_users), cols].sort_values(["cluster", "period"]))"""),
    code("""plotter.plot_clusters(feats, outdir_)
plt.show()"""),
    md("""## 4. Hold-out evaluation (July, unseen users)

The model is compared with two naive forecasts: **last month** (persistence) and the **3-month mean**.

**What to expect, and how to tell if something is wrong:**
* Income is lumpy (salary or nothing; many users have no inflow in a given month), so R² around 0.3–0.4 is
  realistic. Expenses are smoother, so R² is around 0.5.
* The model should clearly beat *last month* on every metric. Against the *3-month mean* it should win on
  RMSE and R², since a squared-error model estimates the conditional **mean**. On income MAE it can lose,
  because MAE rewards the conditional **median** and the median of a lumpy income is often 0.
* Warning signs: R² > 0.9 (leakage, e.g. July data in the features), the model losing to *last month*
  (broken features or targets), or residuals far from 0 (biased model)."""),
    code("""m = metrics_table(holdout)
display(m.pivot(index="method", columns="target", values=["MAE", "RMSE", "R2"]).round(2))
display(pd.DataFrame([bootstrap_mae_diff(holdout, t) for t in ("income", "expenses")]).round(1))"""),
    code("""plotter.plot_evaluation(holdout, outdir_)
plt.show()
display(per_cluster_table(holdout).round(1))"""),
    md("""### Toy example on the hold-out set: one unseen user per cluster

**What to expect:** forecasts track each user's recent level. Large misses come from one-off events
(a single large transfer) that no history-based model can anticipate."""),
    code("""hold_toy = holdout.groupby("cluster").apply(
    lambda g: g.iloc[(g.actual_expenses - g.actual_expenses.median()).abs().argsort()[:1]], include_groups=False)
display(hold_toy[["user_id", "income", "actual_income", "predicted_income", "mean_3m_income",
                  "expenses", "actual_expenses", "predicted_expenses", "mean_3m_expenses"]]
        .rename(columns={"income": "june_income", "expenses": "june_expenses"}))"""),
    md("""## 5. What drives the forecasts

**What to expect:** recent levels dominate (`*_roll3_mean`, current-month `income`/`expenses`). Category
spending averages add little beyond the totals."""),
    code("""plotter.plot_feature_importance(bundle, outdir_)
plt.show()
plotter.plot_shap_summary(bundle, feats, outdir_)
plt.show()"""),
    md("""## 6. Production forecast: August 2016 for every user

Same call as `income predict` and `POST /predict`. Breakdowns split each total by the user's last 3 months
of category shares, so they sum to the total."""),
    code("""aug = predict_next_month(bundle, tx)
print(aug.forecast_month.iloc[0], "forecasts for", len(aug), "users")
display(aug[aug.user_id.isin(toy_users)][["user_id", "cluster", "predicted_income", "predicted_expenses",
                                          "predicted_expenses_breakdown"]])
display(aug[["predicted_income", "predicted_expenses"]].describe().T)"""),
    md("""## Limitations
* Six months of history (Feb–Jul 2016). No seasonality can be learned, and the evaluation covers a single month.
* One hold-out month means wide uncertainty. Use the bootstrap intervals above, not point estimates.
* Breakdowns are proportional allocations, not separate category models."""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("analysis.ipynb").write_text(nbf.writes(nb))
print("wrote analysis.ipynb")
