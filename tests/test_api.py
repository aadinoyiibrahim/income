import pytest
from src.api import app
from fastapi.testclient import TestClient
import pandas as pd
from io import BytesIO

def test_health_endpoint(api_client):
    """Tests the basic /health GET endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.usefixtures("create_dummy_models")
def test_predict_endpoint_success(api_client, dummy_prediction_csv_content):
    """Tests the /predict POST endpoint with a mock CSV upload."""
    files = {
        'file': ('test_transactions.csv', BytesIO(dummy_prediction_csv_content), 'text/csv')
    }
    
    response = api_client.post("/predict", files=files)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code} with response: {response.text}"
    
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) > 0

    first_pred = predictions[0]
    required_keys = ['user_id', 'predicted_income', 'predicted_expenses', 'predicted_income_breakdown', 'predicted_expenses_breakdown']
    assert all(key in first_pred for key in required_keys)
    assert isinstance(first_pred['user_id'], str)
    assert isinstance(first_pred['predicted_income'], (float, int))
    assert isinstance(first_pred['predicted_expenses'], (float, int))
    assert isinstance(first_pred['predicted_income_breakdown'], dict)
    assert isinstance(first_pred['predicted_expenses_breakdown'], dict)
    
    for pred in predictions:
        assert pred['predicted_income'] >= 0
        assert pred['predicted_expenses'] >= 0

def test_predict_endpoint_no_file(api_client):
    """Tests /predict when no file is uploaded."""
    response = api_client.post("/predict")
    assert response.status_code == 422