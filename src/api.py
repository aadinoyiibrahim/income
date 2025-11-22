from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from io import StringIO
from src.predict import predict_and_aggregate

app = FastAPI(
    title="Income Predictor API",
    version="1.0.0",
    description="ML Pipeline for forecasting user income and expenses."
)


class PredictionResponse(BaseModel):
    user_id: str
    predicted_income: float
    predicted_expenses: float
    predicted_income_breakdown: Dict[str, float]
    predicted_expenses_breakdown: Dict[str, float]

@app.post("/predict", response_model=List[PredictionResponse])
async def predict(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    """
    Processes an uploaded CSV file of transactions, predicts next-month income and expenses
    for each user, and returns the results.

    Args:
        file: The uploaded transaction CSV file. Must contain a 'date' column.

    Returns:
        A list of dictionaries, where each dictionary contains the user_id,
        predicted_income, predicted_expenses, and their respective category breakdowns.
    """

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')), parse_dates=['transaction_date'])
        
        required_cols = {'user_id', 'transaction_date', 'transaction_type', 'amount_n26_currency'}
        if not required_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"Missing columns. Required: {required_cols}")

        predictions = predict_and_aggregate(df_agg=None, test_path=None) 
        return predictions.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "version": "1.0.0"}

