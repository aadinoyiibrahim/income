import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Dict, List, Optional, Any

def load_lookups() -> Tuple[Dict[str, str], Dict[float, str]]:
    """Load lookup tables for transaction types and MCC groups."""
    try:
        type_df = pd.read_csv('../data/transaction_types.csv')
        type_to_flow = dict(zip(type_df['type'], type_df['direction']))
        type_to_flow = {k: v for k, v in type_to_flow.items() if v in ['In', 'Out']}  
        mcc_df = pd.read_csv('../data/mcc_group_definition.csv')
        mcc_to_cat = dict(zip(mcc_df['mcc_group'], mcc_df['explanation']))
        return type_to_flow, mcc_to_cat
    except FileNotFoundError:
        print("Error: Lookup files (mcc_group_definition.csv or transaction_types.csv) not found.")
        raise


def preprocess_data(df: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit: bool = False, save_path: Optional[str] = None, fixed_categories: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """
    Preprocess transaction data: map lookups, add temporal features, aggregate, engineer features, and scale.

    Args:
        df: Input DataFrame with transaction-level data.
        scaler: The StandardScaler instance to use for transformation when fit=False. Defaults to None.
        fit: If True, fit a new StandardScaler; otherwise, use the provided scaler. Defaults to False.
        save_path: Full path to save the preprocessed DataFrame. The directory is used to save the scaler and feature columns. Defaults to None.
        fixed_categories: Fixed list of expense categories to ensure consistent features. Defaults to None.
    Returns:
        tuple: (features, processed DataFrame, scaler)
    """
    print("Starting preprocessing...")


    if not {'transaction_type', 'amount_n26_currency', 'user_id', 'transaction_date'}.issubset(df.columns):
        raise ValueError(
            "Input DataFrame must contain 'transaction_type', 'amount_n26_currency', 'user_id', and 'transaction_date' columns.")

    df = df.copy()
    type_to_flow, mcc_to_cat = load_lookups()
    df['flow'] = df['transaction_type'].map(type_to_flow)
    if df['flow'].isna().any():
        print(
            f"Warning: {df['flow'].isna().sum()} transactions have unmapped transaction_type values: {df[df['flow'].isna()]['transaction_type'].unique()}")
        df = df.dropna(subset=['flow'])
    df['category'] = df['mcc_group'].replace('NA', np.nan).astype(float, errors='ignore').map(mcc_to_cat)
    df['category'] = df['category'].fillna('Unknown')

    df['date'] = pd.to_datetime(df['transaction_date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    agg_dict = {
        'month': 'first',
        'amount_n26_currency': ['sum', 'count'],
        'flow': lambda x: (x == 'In').mean()
    }
    if 'cluster' in df.columns:
        agg_dict['cluster'] = 'first'
    else:
        print("Warning: 'cluster' column not found. Proceeding without clustering.")

    agg_data = df.groupby(['user_id', df['date'].dt.to_period('M')]).agg(agg_dict).reset_index()

    agg_columns = ['user_id', 'month_period', 'month', 'total_amount', 'tx_count', 'income_ratio']
    if 'cluster' in agg_dict:
        agg_columns.append('cluster')
    agg_data.columns = agg_columns

    agg_data['income'] = agg_data.apply(
        lambda row: df[(df['user_id'] == row['user_id']) &
                       (df['date'].dt.to_period('M') == row['month_period']) &
                       (df['flow'] == 'In')]['amount_n26_currency'].sum(),
        axis=1
    )
    agg_data['expenses'] = agg_data.apply(
        lambda row: df[(df['user_id'] == row['user_id']) &
                       (df['date'].dt.to_period('M') == row['month_period']) &
                       (df['flow'] == 'Out')]['amount_n26_currency'].sum(),
        axis=1
    )

    print("--5. Computing category breakdowns...")
    agg_data['income_breakdown'] = agg_data.apply(
        lambda row: df[(df['user_id'] == row['user_id']) &
                       (df['date'].dt.to_period('M') == row['month_period']) &
                       (df['flow'] == 'In')].groupby('category')['amount_n26_currency'].sum().to_dict(),
        axis=1
    )
    agg_data['expenses_breakdown'] = agg_data.apply(
        lambda row: df[(df['user_id'] == row['user_id']) &
                       (df['date'].dt.to_period('M') == row['month_period']) &
                       (df['flow'] == 'Out')].groupby('category')['amount_n26_currency'].sum().to_dict(),
        axis=1
    )

    agg_df = agg_data
    agg_df['date'] = agg_df['month_period'].dt.to_timestamp()

    def user_features(user_df: pd.DataFrame) -> pd.DataFrame:
        user_df = user_df.sort_values('month')
        user_df['prev_income'] = user_df['income'].shift(1).fillna(0)
        user_df['prev_expenses'] = user_df['expenses'].shift(1).fillna(0)
        user_df['income_trend'] = (user_df['income'] - user_df['prev_income']) / (user_df['prev_income'] + 1e-6)
        user_df['expenses_trend'] = (user_df['expenses'] - user_df['prev_expenses']) / (user_df['prev_expenses'] + 1e-6)
        categories = fixed_categories if fixed_categories is not None else list(
            user_df['expenses_breakdown'].iloc[0].keys())
        for cat in categories:
            user_df[f'avg_exp_{cat}'] = user_df['expenses_breakdown'].apply(lambda d: d.get(cat, 0)).rolling(
                3).mean().fillna(0)
        return user_df

    agg_df = agg_df.groupby('user_id').apply(user_features, include_groups=True).reset_index(drop=True)

    if 'user_id' not in agg_df.columns:
        raise ValueError("'user_id' column missing after feature engineering.")

    num_cols = [col for col in agg_df.columns if
                col.startswith('prev_') or col.startswith('trend_') or col.startswith('avg_')]
    if not num_cols:
        raise ValueError("No numerical features generated for scaling.")
    if fit:
        scaler = StandardScaler()
        features = scaler.fit_transform(agg_df[num_cols])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(scaler, os.path.join(os.path.dirname(save_path), 'scaler.pkl'))
            print(f"Saved scaler to: {os.path.join(os.path.dirname(save_path), 'scaler.pkl')}")
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False.")
        features = scaler.transform(agg_df[num_cols])

    if save_path and fit:
        joblib.dump(num_cols, os.path.join(os.path.dirname(save_path), 'feature_columns.pkl'))
        agg_df.to_pickle(save_path)
        print(f"Saved preprocessed DataFrame to: {save_path}")

    print("Columns in agg_df after preprocessing:", agg_df.columns)
    print(f"Shape of features: {features.shape}")
    print(f"Shape of agg_df: {agg_df.shape}")

    return features, agg_df, scaler