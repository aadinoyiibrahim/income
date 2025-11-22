import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.cluster import cluster_users
from src.predict import predict_and_aggregate
from src.preprocessing import preprocess_data

@pytest.mark.usefixtures("create_dummy_models")
def test_cluster_users_prediction(dummy_transaction_df):
    """Tests user clustering using a pre-saved dummy model."""
    
    features, agg_df, _ = preprocess_data(dummy_transaction_df, fit=True)
    clustered_df, kmeans_model = cluster_users(dummy_transaction_df, fit=False, model_path='models/kmeans.pkl')

    assert 'cluster' in clustered_df.columns
    assert np.issubdtype(clustered_df['cluster'].dtype, np.integer)
    assert kmeans_model is not None
    assert clustered_df['cluster'].nunique() == 1


@pytest.mark.usefixtures("create_dummy_models")
def test_predict_and_aggregate_end_to_end(dummy_transaction_df):
    """
    Tests the main prediction function end-to-end, leveraging the dummy models.
    This simulates the final prediction step.
    """
    features, agg_df, _ = preprocess_data(dummy_transaction_df, fit=True)
    predictions_df = predict_and_aggregate(df_agg=agg_df, features=features)
    assert not predictions_df.empty

    assert isinstance(predictions_df, pd.DataFrame)
    assert predictions_df.shape[0] > 0
    
    required_cols = {
        'user_id',
        'predicted_income',
        'predicted_expenses',
        'predicted_income_breakdown',
        'predicted_expenses_breakdown',
    }
    
    assert all(col in predictions_df.columns for col in required_cols)
    
    for _, row in predictions_df.iterrows():
            assert row["predicted_income"] > 0
            assert row["predicted_expenses"] > 0

            income_sum = sum(row["predicted_income_breakdown"].values())
            expenses_sum = sum(row["predicted_expenses_breakdown"].values())

            assert np.isclose(income_sum, row["predicted_income"], atol=0.1)
            assert np.isclose(expenses_sum, row["predicted_expenses"], atol=0.1)
