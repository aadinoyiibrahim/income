import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
from preprocessing import load_lookups 
from typing import Tuple, Dict, Any, List

def cluster_users(df: pd.DataFrame, model_path: str ='../models/kmeans.pkl', fit: bool =False) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster users based on aggregated features (avg/total amount, tx count, income ratio).

    Args:
        df: Input DataFrame with transactions (must include 'user_id', 'amount_n26_currency',
            'transaction_type', 'mcc_group').
        model_path: Path to save/load the KMeans model. Defaults to './models/kmeans.pkl'.
        fit: If True, fit a new KMeans model; otherwise, load the existing model. Defaults to False.

    Returns:
        A tuple containing:
        - pd.DataFrame: The original DataFrame merged with the 'cluster' column.
        - KMeans: The fitted or loaded KMeans model.

    Raises:
        ValueError: If the input DataFrame is missing required columns.
    """
    print("Starting clustering...")

    required_cols = ['user_id', 'amount_n26_currency', 'transaction_type', 'mcc_group']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

    df = df.copy()
    type_to_flow, mcc_to_cat = load_lookups()

    if 'flow' not in df.columns:
        print("Mapping flow column...")
        df['flow'] = df['transaction_type'].map(type_to_flow)
        if df['flow'].isna().any():
            print(
                f"Warning: {df['flow'].isna().sum()} transactions have unmapped transaction_type values: {df[df['flow'].isna()]['transaction_type'].unique()}")
            df = df.dropna(subset=['flow'])

    if 'category' not in df.columns:
        print("Mapping category column...")
        df['category'] = df['mcc_group'].replace('NA', np.nan).astype(float, errors='ignore').map(mcc_to_cat)
        df['category'] = df['category'].fillna('Unknown')


    print("Aggregating data per user...")
    agg_dict = {
        'amount_n26_currency': ['mean', 'sum', 'count'],
        'flow': lambda x: (x == 'In').mean()  # Proportion of 'In' transactions
    }
    try:
        user_agg = df.groupby('user_id').agg(agg_dict).reset_index()
        user_agg.columns = ['user_id', 'avg_amount', 'total_amount', 'tx_count', 'income_ratio']
    except Exception as e:
        print(f"Error in aggregation: {e}. Using fallback without flow.")
        agg_dict.pop('flow', None)
        user_agg = df.groupby('user_id').agg(agg_dict).reset_index()
        user_agg.columns = ['user_id', 'avg_amount', 'total_amount', 'tx_count']
        user_agg['income_ratio'] = 0  # Fallback for missing flow


    print("Performing clustering...")
    X = user_agg[['avg_amount', 'total_amount', 'tx_count', 'income_ratio']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if fit:
        kmeans = KMeans(n_clusters=5, random_state=42)
        user_agg['cluster'] = kmeans.fit_predict(X_scaled)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(kmeans, model_path)
        print(f"Saved KMeans model to: {model_path}")
    else:
        try:
            kmeans = joblib.load(model_path)
            user_agg['cluster'] = kmeans.predict(X_scaled)
        except FileNotFoundError:
            print(f"Error: KMeans model not found at {model_path}. Fitting new model.")
            kmeans = KMeans(n_clusters=5, random_state=42)
            user_agg['cluster'] = kmeans.fit_predict(X_scaled)
            joblib.dump(kmeans, model_path)
            print(f"Saved new KMeans model to: {model_path}")

 
    df = df.merge(user_agg[['user_id', 'cluster']], on='user_id', how='left')
    print(f"Shape of clustered DataFrame: {df.shape}")
    print("Clustering completed.")
    return df, kmeans