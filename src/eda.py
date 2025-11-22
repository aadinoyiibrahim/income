import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ks_2samp, chi2_contingency
from preprocessing import load_lookups
from typing import Optional, Tuple, Dict, Any, List

def perform_eda(data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Perform exploratory data analysis: summaries, statistical tests, and visualizations.

    Plots are generated and saved to the './img/eda' directory.

    Args:
        data_path: Path to raw transaction CSV. If None, the provided df is used. Defaults to None.
        df: Preprocessed DataFrame with transaction data. If provided, data_path is ignored. Defaults to None.

    Returns:
        The processed DataFrame, guaranteed to include 'flow' and 'category' columns.

    Raises:
        ValueError: If neither data_path nor df is provided, or if df is missing required columns.
        FileNotFoundError: If data_path is provided but the file is not found.
    """
    print("Starting EDA...")

    
    if data_path and df is None:
        print(f"Loading data from: {data_path}")
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Data file {data_path} not found.")
            raise
  
        print("Mapping flow and category...")
        type_to_flow, mcc_to_cat = load_lookups()
        df['flow'] = df['transaction_type'].map(type_to_flow)
        if df['flow'].isna().any():
            print(f"Warning: {df['flow'].isna().sum()} transactions have unmapped transaction_type values.")
            df = df.dropna(subset=['flow'])
        df['category'] = df['mcc_group'].replace('NA', np.nan).astype(float, errors='ignore').map(mcc_to_cat)
        df['category'] = df['category'].fillna('Unknown')
    
    elif df is not None:
        print("Using provided DataFrame for EDA.")
        required_cols = ['user_id', 'transaction_date', 'amount_n26_currency', 'flow', 'category']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
    
    else:
        raise ValueError("Either data_path or df must be provided.")


    df = df.copy()
    df['date'] = pd.to_datetime(df['transaction_date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)


    print("Generating summaries...")
    print(f"Number of transactions: {len(df)}")
    print(f"Number of unique users: {df['user_id'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Flow distribution: {df['flow'].value_counts(normalize=True).to_dict()}")
    print(f"Category distribution: {df['category'].value_counts(normalize=True).to_dict()}")
    print(f"Missing MCC groups: {df['mcc_group'].eq('NA').sum()} ({df['mcc_group'].eq('NA').mean():.1%})")


    print("Running statistical tests...")
    in_amounts = df[df['flow'] == 'In']['amount_n26_currency']
    out_amounts = df[df['flow'] == 'Out']['amount_n26_currency']
    ks_stat, ks_p = ks_2samp(in_amounts, out_amounts)
    print(f"Kolmogorov-Smirnov test (In vs Out amounts): KS Stat = {ks_stat:.4f}, p-value = {ks_p:.4f}")


    contingency_table = pd.crosstab(df['flow'], df['category'])
    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square test (flow vs category): Chi2 Stat = {chi2_stat:.1f}, p-value = {chi2_p:.4f}")


    contingency_table_month = pd.crosstab(df['transaction_type'], df['month'])
    chi2_stat_month, chi2_p_month, _, _ = chi2_contingency(contingency_table_month)
    print(
        f"Chi-Square test (transaction_type vs month): Chi2 Stat = {chi2_stat_month:.1f}, p-value = {chi2_p_month:.4f}")


    print("Generating visualizations...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.ravel()

    # ----------
    # plots
    # -----------
    sns.histplot(df['amount_n26_currency'], bins=50, ax=axes[0])
    axes[0].set_title('Amount Distribution')


    sns.boxplot(x='flow', y='amount_n26_currency', data=df, ax=axes[1])
    axes[1].set_title('Amount by Flow')

    sns.boxplot(y='category', x='amount_n26_currency', data=df, ax=axes[2])
    axes[2].set_title('Amount by Category')

    sns.countplot(x='month', data=df, ax=axes[3])
    axes[3].set_title('Transaction Count by Month')

    sns.countplot(x='flow', data=df, ax=axes[4])
    axes[4].set_title('Flow Distribution')

    sns.countplot(y='category', data=df, ax=axes[5])
    axes[5].set_title('Category Distribution')

    monthly_totals = df.groupby(df['date'].dt.to_period('M'))['amount_n26_currency'].sum()
    monthly_totals.plot(kind='line', marker='o', ax=axes[6])
    axes[6].set_title('Monthly Total Amount')

    flow_trend = df.groupby(['month', 'flow'])['amount_n26_currency'].sum().unstack()
    flow_trend.plot(kind='line', ax=axes[7])
    axes[7].set_title('Income vs Expenses Trend')

    if 'cluster' in df.columns:
        sns.countplot(x='cluster', data=df, ax=axes[8])
        axes[8].set_title('Cluster Distribution')
    else:
        axes[8].axis('off')
        axes[8].text(0.5, 0.5, 'No Cluster Data', ha='center', va='center')

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[9])
    axes[9].set_title('Correlation Heatmap')

    df['z_score'] = (df['amount_n26_currency'] - df['amount_n26_currency'].mean()) / df['amount_n26_currency'].std()
    anomalies = df[df['z_score'].abs() > 3]
    sns.scatterplot(x='date', y='amount_n26_currency', hue=df['z_score'].abs() > 3, data=df, ax=axes[10])
    axes[10].set_title('Anomalies (Z-score > 3)')

    user_agg = df.groupby('user_id').agg({'amount_n26_currency': 'sum', 'transaction_type': 'count'})
    sns.scatterplot(x='transaction_type', y='amount_n26_currency', data=user_agg, ax=axes[11])
    axes[11].set_title('Transaction Count vs Total Amount')

    plt.tight_layout()
    os.makedirs('../img/eda', exist_ok=True)
    plt.savefig('../img/eda/eda_grid_detailed.png')
    plt.close()

    print("EDA completed. Plots saved to ../img/eda/")
    return df