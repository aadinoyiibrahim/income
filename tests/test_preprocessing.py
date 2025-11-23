import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from src.preprocessing import load_lookups, preprocess_data

def test_load_lookups_success():
    """Tests if lookup files are loaded and correctly mapped."""
    type_to_flow, mcc_to_cat = load_lookups()
    
    assert 'T1' in type_to_flow
    assert type_to_flow['T1'] == 'In'
    assert 'T4' in type_to_flow
    assert type_to_flow['T4'] == 'Out'
    assert 'T5' not in type_to_flow, "Should filter out '-' direction"
    assert 1000 in mcc_to_cat
    assert mcc_to_cat[1000] == 'Groceries'
    assert 4000 in mcc_to_cat
    assert mcc_to_cat[4000] == 'Unknown'

def test_preprocess_data_feature_creation(dummy_transaction_df):
    """Tests if aggregation, temporal features, and category columns are correctly created."""
    features, df_agg, _ = preprocess_data(dummy_transaction_df, fit=True)
    expected_cols = ['user_id', 'month', 'income', 'expenses', 'income_breakdown', 'expenses_breakdown']
    assert all(col in df_agg.columns for col in expected_cols)
    
    feature_cols = [col for col in df_agg.columns if col.startswith('prev_') or col.startswith('avg_')]
    assert len(feature_cols) > 0, "Feature engineering should create lagged columns."

    u1_july = df_agg[(df_agg['user_id'] == 'u1') & (df_agg['month'] == 7)]
    assert u1_july.shape[0] == 1
    assert u1_july['income'].iloc[0] == 300.0
    assert u1_july['expenses'].iloc[0] == 0.0
    assert 'Unknown' in u1_july['income_breakdown'].iloc[0]
    
    u1_july_agg = df_agg.set_index(['user_id', 'month']).loc[('u1', 7)]
    assert pytest.approx(u1_july_agg['prev_income'], 0.01) == 100.0
    assert pytest.approx(u1_july_agg['prev_expenses'], 0.01) == -20.0
    

def test_preprocess_data_scaling(dummy_transaction_df):
    """Tests the scaling of numerical features."""
    features_fit, df_agg_fit, _ = preprocess_data(dummy_transaction_df, fit=True, save_path='models/test_agg_df.pkl')

    import joblib
    scaler = joblib.load('models/scaler.pkl')
    features_transform, df_agg_transform, _ = preprocess_data(dummy_transaction_df, scaler=scaler, fit=False)
    
    assert features_transform.shape[0] == df_agg_transform.shape[0]
    assert features_transform.ndim == 2
