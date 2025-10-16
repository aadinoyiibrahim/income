from fastapi import FastAPI, UploadFile, File
from io import StringIO
import pandas as pd
import uvicorn
from predict import predict_and_aggregate

app = FastAPI(title="Income Predictor API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')), parse_dates=['date'])
    predictions = predict_and_aggregate(df)  # Overload for DF input
    return predictions.to_dict(orient='records')

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)