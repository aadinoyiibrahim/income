import os
from gensim import corpora
from gensim.models import LdaModel
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress FutureWarning for groupby.apply

def infer_subcategories(df):
    """Infer subcategories using LDA on descriptions (placeholder if no descriptions)."""
    texts = [str(cat).split() for cat in df['category']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    df['subcategory'] = [max(lda[corpus[i]], key=lambda x: x[1])[0] for i in range(len(texts))]
    topic_map = {0: 'Salary', 1: 'Rent', 2: 'Groceries', 3: 'Utilities', 4: 'Other'}
    df['subcategory'] = df['subcategory'].map(topic_map)
    return df

def predict_and_aggregate(df_agg=None, features=None, test_path=None, models_dir='../models'):
    """
    Predict next-month income and expenses for users, aggregating results with category breakdowns.
    
    Args:
        df_agg (pd.DataFrame, optional): Preprocessed DataFrame with aggregated user-month data.
        features (np.ndarray, optional): Preprocessed feature array for df_agg.
        test_path (str, optional): Path to test CSV. If None, use holdout_df.pkl.
        models_dir (str): Directory containing models and holdout data.
    
    Returns:
        pd.DataFrame: Predictions with user_id, predicted_income, predicted_expenses, and breakdowns.
    """
    if test_path:
        print("Loading test data from:", test_path)
        df = pd.read_csv(test_path)
        # Preprocessing should be done externally in analysis.ipynb
        raise ValueError("Test data should be preprocessed and passed as df_agg and features.")
    
    if df_agg is None or features is None:
        print("Loading holdout data from:", os.path.join(models_dir, 'holdout_df.pkl'))
        try:
            df_agg = joblib.load(os.path.join(models_dir, 'holdout_df.pkl'))
        except FileNotFoundError:
            print("Error: Holdout file not found.")
            raise
        print("Columns in holdout_df:", df_agg.columns)
        print("Applying feature engineering to holdout data...")
        def user_features(user_df):
            user_df = user_df.sort_values('month')
            user_df['prev_income'] = user_df['income'].shift(1).fillna(0)
            user_df['prev_expenses'] = user_df['expenses'].shift(1).fillna(0)
            user_df['income_trend'] = (user_df['income'] - user_df['prev_income']) / (user_df['prev_income'] + 1e-6)
            user_df['expenses_trend'] = (user_df['expenses'] - user_df['prev_expenses']) / (user_df['prev_expenses'] + 1e-6)
            # Load fixed categories from training
            try:
                train_num_cols = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
                fixed_categories = [col.replace('avg_exp_', '') for col in train_num_cols if col.startswith('avg_exp_')]
            except FileNotFoundError:
                fixed_categories = list(user_df['expenses_breakdown'].iloc[0].keys())
            for cat in fixed_categories:
                user_df[f'avg_exp_{cat}'] = user_df['expenses_breakdown'].apply(lambda d: d.get(cat, 0)).rolling(3).mean().fillna(0)
            return user_df
        df_agg = df_agg.groupby('user_id').apply(user_features, include_groups=False).reset_index()
        # Scale features
        print("Scaling features for holdout data...")
        num_cols = [col for col in df_agg.columns if col.startswith('prev_') or col.startswith('trend_') or col.startswith('avg_')]
        try:
            scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        except FileNotFoundError:
            print("Error: Scaler file not found.")
            raise
        features = scaler.transform(df_agg[num_cols])
        # Align features with training columns
        try:
            train_num_cols = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
            if set(num_cols) != set(train_num_cols):
                print(f"Feature mismatch: holdout has {len(num_cols)} features, training has {len(train_num_cols)}")
                # Add missing columns with zeros
                missing_cols = [col for col in train_num_cols if col not in num_cols]
                if missing_cols:
                    print(f"Adding missing columns: {missing_cols}")
                    for col in missing_cols:
                        df_agg[col] = 0
                    num_cols = train_num_cols
                    features = scaler.transform(df_agg[num_cols])
                # Remove extra columns if any
                extra_cols = [col for col in num_cols if col not in train_num_cols]
                if extra_cols:
                    print(f"Removing extra columns: {extra_cols}")
                    df_agg = df_agg.drop(columns=extra_cols)
                    num_cols = [col for col in num_cols if col not in extra_cols]
                    features = scaler.transform(df_agg[num_cols])
        except FileNotFoundError:
            print("Warning: feature_columns.pkl not found. Proceeding without feature alignment.")
        print(f"Shape of features after scaling: {features.shape}")
    
    # Validate input DataFrame
    required_cols = ['user_id', 'month', 'income', 'expenses', 'cluster']
    if not all(col in df_agg.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_agg.columns]
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
    
    # For holdout, use last month's data for prediction
    print("Selecting last month for prediction...")
    last_month = df_agg.groupby('user_id').last().reset_index()
    print(f"Shape of last_month: {last_month.shape}")
    print(f"Number of user_ids in last_month: {last_month['user_id'].nunique()}")
    
    '''# Align features with last_month
    last_month_indices = df_agg.index[df_agg['user_id'].isin(last_month['user_id'])].values
    if len(last_month_indices) != len(last_month):
        print(f"Warning: Mismatch in number of rows (last_month: {len(last_month)}, indices: {len(last_month_indices)})")
    X_last = features[last_month_indices]
    print(f"Shape of X_last: {X_last.shape}")'''
    # Align features correctly with last_month
    df_agg = df_agg.sort_values(['user_id', 'month']).reset_index(drop=True)
    last_idx = df_agg.groupby('user_id').tail(1).index  # last row per user
    X_last = features[last_idx]
    last_month = df_agg.loc[last_idx].reset_index(drop=True)

    print(f"Shape of last_month: {last_month.shape}")
    print(f"Shape of X_last: {X_last.shape}")

    
    # Load models with fallback
    print("Loading models...")
    models_income = {}
    models_expenses = {}
    available_clusters = []
    for c in range(5):
        income_model_path = os.path.join(models_dir, f'xgb_income_cluster_{c}.pkl')
        expenses_model_path = os.path.join(models_dir, f'xgb_expenses_cluster_{c}.pkl')
        try:
            models_income[c] = joblib.load(income_model_path)
            models_expenses[c] = joblib.load(expenses_model_path)
            print(f"Loaded models for cluster {c}")
            available_clusters.append(c)
        except FileNotFoundError:
            print(f"Warning: Models for cluster {c} not found at {income_model_path} or {expenses_model_path}. Skipping cluster.")
    
    if not available_clusters:
        print("Error: No models found for any clusters. Cannot proceed with predictions.")
        raise FileNotFoundError("No valid model files found in ../models/.")
    
    # Predict
    print("Making predictions...")
    last_month['cluster'] = last_month['cluster'].fillna(available_clusters[0]).astype(int)
    pred_income = np.zeros(len(last_month))
    pred_expenses = np.zeros(len(last_month))
    for cluster in last_month['cluster'].unique():
        if cluster not in available_clusters:
            print(f"Warning: No model for cluster {cluster}. Using model for cluster {available_clusters[0]}.")
            cluster = available_clusters[0]
        mask = last_month['cluster'] == cluster
        if mask.sum() == 0:
            print(f"Warning: No data for cluster {cluster}. Skipping.")
            continue
        # Use X_last[mask] directly for prediction (aligned with last_month)
        pred_income[mask] = models_income[cluster].predict(X_last[mask])
        pred_expenses[mask] = models_expenses[cluster].predict(X_last[mask])
    
    last_month['predicted_income'] = pred_income
    last_month['predicted_expenses'] = pred_expenses
    
    # Allocate category breakdowns
    print("Allocating category breakdowns...")
    def allocate_breakdown(row, is_income):
        hist = row['income_breakdown'] if is_income else row['expenses_breakdown']
        total = row['predicted_income'] if is_income else row['predicted_expenses']
        if sum(hist.values()) == 0:
            return {}
        prop = {k: v / sum(hist.values()) for k, v in hist.items()}
        return {k: v * total for k, v in prop.items()}
    
    last_month['predicted_income_breakdown'] = last_month.apply(allocate_breakdown, axis=1, is_income=True)
    last_month['predicted_expenses_breakdown'] = last_month.apply(allocate_breakdown, axis=1, is_income=False)
    
    # Final output
    agg = last_month[['user_id', 'predicted_income', 'predicted_expenses', 'predicted_income_breakdown', 'predicted_expenses_breakdown']]
    os.makedirs('../result', exist_ok=True)
    agg.to_csv(os.path.join('../result', 'august_predictions.csv'), index=False)
    print("Predictions saved to: ../result/august_predictions.csv")
    return agg