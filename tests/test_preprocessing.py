import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    add_features,
    add_targets,
    annotate_transactions,
    build_feature_frame,
    monthly_panel,
    parse_dates,
    validate_transactions,
)


def test_parse_dates_handles_both_shipped_formats():
    s = pd.Series(["2016-02-01", "01.02.16", "31.05.16"])
    assert list(parse_dates(s).dt.strftime("%Y-%m-%d")) == ["2016-02-01", "2016-02-01", "2016-05-31"]


def test_parse_dates_rejects_garbage():
    with pytest.raises(ValueError):
        parse_dates(pd.Series(["not a date"]))


def test_validate_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_transactions(pd.DataFrame({"user_id": ["a"]}))


def test_lookups_filter_non_monetary_types(lookups):
    assert lookups.type_to_flow["CT"] == "In" and lookups.type_to_flow["PT"] == "Out"
    assert "AR" not in lookups.type_to_flow  # direction "-": authorisation reject
    assert lookups.mcc_to_category[1] == "ATM" and "Unknown" in lookups.categories


def _tiny():
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "a"],
            "transaction_date": pd.to_datetime(["2016-02-03", "2016-02-10", "2016-04-01", "2016-04-05", "2016-02-11"]),
            "transaction_type": ["CT", "PT", "PT", "DD", "AR"],
            "mcc_group": [np.nan, 2, 1, np.nan, 2],
            "amount_n26_currency": [1000, 50, 20, 30, 999],
        }
    )


def test_panel_fills_inactive_months_with_zero(lookups):
    tx = annotate_transactions(_tiny(), lookups)
    assert "AR" not in set(tx["transaction_type"])  # dropped
    panel = monthly_panel(tx, lookups.categories)
    a = panel[panel.user_id == "a"].set_index("period")
    assert list(a.index.astype(str)) == ["2016-02", "2016-03", "2016-04"]
    assert a.loc[pd.Period("2016-03", "M"), ["income", "expenses", "tx_count"]].tolist() == [0, 0, 0]
    assert a.loc[pd.Period("2016-02", "M"), "income"] == 1000
    assert a.loc[pd.Period("2016-02", "M"), "exp_Groceries"] == 50
    # b starts in April: no fabricated history before its first month
    assert (panel.user_id == "b").sum() == 1


def test_lags_and_targets(lookups):
    panel = monthly_panel(annotate_transactions(_tiny(), lookups), lookups.categories)
    f = add_targets(add_features(panel, lookups.categories))
    a = f[f.user_id == "a"].reset_index(drop=True)
    assert a.loc[1, "income_lag1"] == 1000 and a.loc[2, "income_lag2"] == 1000
    assert a.loc[0, "target_expenses"] == 0 and a.loc[1, "target_expenses"] == 20
    assert np.isnan(a.loc[2, "target_income"])
    assert a.loc[2, "expenses_roll3_mean"] == pytest.approx((50 + 0 + 20) / 3)


def test_features_do_not_look_ahead(transactions, lookups):
    """Features at month t must be identical whether or not later months are present."""
    early = transactions[transactions.transaction_date < "2016-04-01"]
    f_early = build_feature_frame(early, lookups)
    f_full = build_feature_frame(transactions, lookups)
    cols = [c for c in f_early.columns if c not in ("user_id", "period")]
    key = ["user_id", "period"]
    merged = f_early.merge(f_full, on=key, suffixes=("_e", "_f"))
    for c in cols:
        np.testing.assert_allclose(merged[f"{c}_e"], merged[f"{c}_f"], err_msg=c)
