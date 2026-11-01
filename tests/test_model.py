import numpy as np
import pandas as pd

from src.bundle import GLOBAL, load_bundle
from src.cluster import assign_clusters, user_profiles
from src.predict import predict_next_month
from src.preprocessing import build_feature_frame


def test_training_reports_model_and_baselines(trained):
    bundle, holdout = trained
    methods = {(r["target"], r["method"]) for r in bundle.metrics["overall"]}
    assert {("income", "model"), ("expenses", "mean_3m"), ("income", "last_month")} <= methods
    assert holdout["forecast_month"].unique().tolist() == ["2016-05"]
    assert set(bundle.models) == {"income", "expenses"}
    for t in bundle.models:  # both targets get the same clusters (old code lost expenses models)
        assert GLOBAL in bundle.models[t]
    assert set(bundle.models["income"]) == set(bundle.models["expenses"])


def test_model_beats_persistence_on_structured_data(trained):
    m = pd.DataFrame(trained[0].metrics["overall"]).set_index(["target", "method"])["MAE"]
    assert m[("expenses", "model")] < m[("expenses", "last_month")]


def test_cluster_assignment_independent_of_batch(trained, transactions, lookups):
    bundle = trained[0]
    prof = user_profiles(build_feature_frame(transactions, lookups))
    all_at_once = assign_clusters(bundle.clusterer, prof)
    one_by_one = pd.concat([assign_clusters(bundle.clusterer, prof.iloc[[i]]) for i in range(5)])
    assert (all_at_once.iloc[:5] == one_by_one).all()


def test_predict_next_month(bundle_path, transactions):
    bundle = load_bundle(bundle_path)
    preds = predict_next_month(bundle, transactions)
    assert preds["forecast_month"].unique().tolist() == ["2016-06"]
    assert len(preds) == transactions["user_id"].nunique()
    assert (preds[["predicted_income", "predicted_expenses"]] >= 0).all().all()
    for _, r in preds.iterrows():
        assert np.isclose(sum(r["predicted_income_breakdown"].values()), r["predicted_income"], atol=0.05)
        assert np.isclose(sum(r["predicted_expenses_breakdown"].values()), r["predicted_expenses"], atol=0.05)


def test_prediction_depends_on_input(bundle_path, transactions):
    bundle = load_bundle(bundle_path)
    one_user = transactions[transactions.user_id == "u000"]
    rich = one_user.assign(amount_n26_currency=one_user["amount_n26_currency"] * 20)
    p1 = predict_next_month(bundle, one_user).iloc[0]
    p2 = predict_next_month(bundle, rich).iloc[0]
    assert (p1["predicted_expenses"], p1["predicted_income"]) != (p2["predicted_expenses"], p2["predicted_income"])
