# Income and Expenses Prediction Pipeline

This project predicts monthly income and expenses for users based on financial transaction data from February to July 2016. 

## Task

Predict income and expenses for a holdout sample of ~10K users for the month of August based on a training sample of ~10K users from Februrary through July.

Based on your judgement of the usefulness of the results, either aggregate the data into incoming & outgoing flows, or predict based on the transaction type/category.

We start by aggregating data into In (incoming) and Out (outgoing) flows. And this would be the response feature.

## what's included

- `src/`:
- `analysis.ipynb`: Main notebook for end-to-end analysis.
- `eda.py`: Exploratory data analysis module.
- `preprocessing.py`: Data cleaning and feature engineering.
- `cluster.py`: User clustering with K-means.
- `train_model.py`: Model training with XGBoost and Optuna.
- `predict.py`: Prediction and aggregation.
- `plotter.py`: Metrics and feature importance plots.
- `api.py`: FastAPI for deployment.
- `data/`: Datasets.
- `models/`: Saved models and scaler.
- `result/`: Predictions CSV.
- `img/`: Plots from EDA and metrics.
- `requirements.txt`: Dependencies.
- `Dockerfile`: For containerization.
- `README.md`: This file.

# How to run the code

## running locally

git clone this repo and `cd income`.
You can install the dependencies needed to run the pipeline using ```pip install -r requirements.txt```. In particular, this script needs to be executed as:

```bash
pip install -r requirements.txt
```
Execute `analysis.ipynb` for full pipeline. Open `src/analysis.ipynb` and run cells sequentially for EDA, clustering, training, prediction.


This will save predictions in `./result/predictions_all_models.csv`.

## API Usage: 
Upload CSV to `/predict` for JSON forecasts.

## running with docker/deployment

Assuming you have docker destop installed. Open the Docker Desktop application. Next, build the docker image by runing the following on your commandline

```bash
docker build --no-cache -t ml_predict_app .
```

and run the container with the following:

```bash
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/result:/app/result \
  ml_predict_app
```

- Build Docker: `docker build -t income-predictor .`
- Run: `docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/result:/app/result income-predictor`
- API Endpoint: POST `/predict` with CSV upload for forecasts.

This will save predictions in `./result/predictions_all_models.csv`. This approach would help you integrate the code into your app.

## EDA
<p align="center" width=10%>
<img src ="./img/eda/eda_grid_detailed.png"><br>
</p>

# Metrics

We consider three metrics; MAE, MSE and R2.

<p align="center" width=10%>
<img src ="./img/metrics_table.png"><br>
</p>

<p align="center" width=10%>
<img src ="./img/cluster_metrics.png"><br>
</p>

<p align="center" width=10%>
<img src ="./img/actual_vs_predicted.png"><br>
</p>

<p align="center" width=10%>
<img src ="./img/shap_summary.png"><br>
</p>


# Contact
You can reachout here [aadinoyiibrahim](https://github.com/aadinoyiibrahim) and RA

## Thanks and Support

[![Thank You Clicks](https://hits.sh/github.com/aadinoyiibrahim/income.svg?label=Thanks%20❤️&color=orange)](https://github.com/aadinoyiibrahim/income)

[![Star on GitHub](https://img.shields.io/github/stars/aadinoyiibrahim/income?style=social)](https://github.com/aadinoyiibrahim/income/stargazers)
