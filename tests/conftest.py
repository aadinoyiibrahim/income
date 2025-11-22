
import sys
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import pytest
import pandas as pd
import numpy as np
import os
import shutil
from fastapi.testclient import TestClient
from src.api import app

TEST_DIRS = ['data', 'models', 'result']
DUMMY_MCC_PATH = 'data/mcc_group_definition.csv'
DUMMY_TYPE_PATH = 'data/transaction_types.csv'

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Sets up the required directories and dummy lookup files before testing."""

    for d in TEST_DIRS:
        os.makedirs(d, exist_ok=True)

    mcc_data = {
        'mcc_group': ['1000', '2000', '3000', '4000'],
        'explanation': ['Groceries', 'Shopping/(Fashion)', 'ATM', 'Unknown']
    }
    mcc_df = pd.DataFrame(mcc_data)
    mcc_df.to_csv(DUMMY_MCC_PATH, index=False)

    type_data = {
        'type': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'direction': ['In', 'Out', 'In', 'Out', '-']
    }
    type_df = pd.DataFrame(type_data)
    type_df.to_csv(DUMMY_TYPE_PATH, index=False)

    yield

    for d in TEST_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d)

@pytest.fixture(scope="session")
def api_client():
    """Fixture to provide a TestClient for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def dummy_transaction_df():
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2', 'u1', 'u2'],
        'transaction_date': pd.to_datetime([
            '2016-06-01', '2016-06-15',
            '2016-06-10', '2016-07-01',
            '2016-07-15', '2016-07-20',
        ]),
        'amount_n26_currency': [100.0, -20.0, 50.0, -10.0, 300.0, -50.0],
        'transaction_type': ['T1', 'T2', 'T3', 'T4', 'T1', 'T2'],
        'mcc_group': ['4000', '1000', '4000', '2000', '4000', '1000'],
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def dummy_prediction_csv_content(dummy_transaction_df):
    """Fixture to provide CSV content bytes for API testing (August predictions)."""
    # Create a small DataFrame that simulates the August test data
    pred_data = {
        'user_id': ['u3', 'u3', 'u4', 'u4'],
        'transaction_date': pd.to_datetime(['2016-08-01', '2016-08-10', '2016-08-05', '2016-08-15']),
        'amount_n26_currency': [500.0, -30.0, 150.0, -70.0],
        'transaction_type': ['T1', 'T2', 'T1', 'T2'],
        'mcc_group': ['4000', '1000', '4000', '1000']
    }
    pred_df = pd.DataFrame(pred_data)
    
    csv_buffer = pd.io.common.StringIO()
    pred_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

@pytest.fixture
def create_dummy_models(dummy_transaction_df):
    """Creates dummy scaler, feature columns list, and XGBoost models."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import joblib
    import xgboost as xgb
    from src.preprocessing import preprocess_data
    # from src.cluster import cluster_users

    scaler = StandardScaler()
    scaler.fit(np.array([[0, 0, 0]]))
    joblib.dump(scaler, 'models/scaler.pkl')

    num_cols = ['prev_income', 'avg_expenses_Groceries', 'tx_count']
    joblib.dump(num_cols, 'models/feature_columns.pkl')

    kmeans = KMeans(n_clusters=1, random_state=42, n_init=1)
    kmeans.fit(np.array([[0, 0, 0, 0]]))
    joblib.dump(kmeans, 'models/kmeans.pkl')

    for flow in ['income', 'expenses']:
        model = xgb.XGBRegressor(random_state=42)
        model.fit(np.array([[10, 20, 30]]), np.array([50]))
        joblib.dump(model, f'models/{flow}_cluster_0.pkl')
    
    features, agg_df, _ = preprocess_data(dummy_transaction_df,fit=True,save_path='models/holdout_df.pkl')
    # agg_df, kmeans = cluster_users(dummy_transaction_df, fit=False, model_path='models/kmeans.pkl') 
    
    # clustered_df, _ = cluster_users(dummy_transaction_df,fit=False,model_path='models/kmeans.pkl')
    # clustered_df.to_pickle('models/holdout_df.pkl')
    
    print("Dummy models and data saved for prediction testing.")
    return True









