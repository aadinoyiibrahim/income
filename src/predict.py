import os
from gensim import corpora
from gensim.models import LdaModel
import joblib
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler 
warnings.filterwarnings('ignore', category=FutureWarning)

def infer_subcategories(df: pd.DataFrame) -> pd.DataFrame:

    """
    Infer subcategories using LDA on transaction categories (placeholder implementation).

    Note: This is a simplified placeholder as transaction descriptions are typically
    used for LDA, but here it uses the 'category' column.

    Args:
        df: Input DataFrame with a 'category' column.

    Returns:
        DataFrame with an added 'subcategory' column.
    """

    texts = [str(cat).split() for cat in df['category']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    df['subcategory'] = [max(lda[corpus[i]], key=lambda x: x[1])[0] for i in range(len(texts))]
    topic_map = {0: 'Salary', 1: 'Rent', 2: 'Groceries', 3: 'Utilities', 4: 'Other'}
    df['subcategory'] = df['subcategory'].map(topic_map)
    return df

def predict_and_aggregate(df_agg: Optional[pd.DataFrame] = None, features: Optional[np.ndarray] = None, test_path: Optional[str] = None, models_dir: str ='./models') -> pd.DataFrame:
    """
    Predict next-month income and expenses for users, aggregating results with category breakdowns.

    If df_input is provided (e.g., from the API upload), it is treated as raw data and fully processed
    before prediction. It then uses the last month of data for prediction.

    Args:
        df_input: The input DataFrame. If None, 'holdout_df.pkl' is loaded.
        features: Preprocessed feature array (ignored if df_input is None).
        test_path: Path to raw test CSV (currently raises error, requires full DF input).
        models_dir: Directory containing models and holdout data. Defaults to './models'.

    Returns:
        A DataFrame containing predictions: 'user_id', 'predicted_income', 'predicted_expenses',
        'predicted_income_breakdown', and 'predicted_expenses_breakdown'.

    Raises:
        ValueError: If test_path is used or if the input DataFrame is missing required columns.
        FileNotFoundError: If required model files are not found.
    """

    if test_path:
        print("Loading test data from:", test_path)
        df = pd.read_csv(test_path)
        raise ValueError("Test data should be preprocessed and passed as df_agg and features.")
    
    if df_agg is None or features is None:
        print("Loading holdout data from:", os.path.join(models_dir, 'holdout_df.pkl'))
        try:
            df_agg = joblib.load(os.path.join(models_dir, 'holdout_df.pkl'))
        except FileNotFoundError:
            print("Error: Holdout file not found.")
            raise

        def user_features(user_df):
            user_df = user_df.sort_values('month')
            user_df['prev_income'] = user_df['income'].shift(1).fillna(0)
            user_df['prev_expenses'] = user_df['expenses'].shift(1).fillna(0)
            user_df['income_trend'] = (user_df['income'] - user_df['prev_income']) / (user_df['prev_income'] + 1e-6)
            user_df['expenses_trend'] = (user_df['expenses'] - user_df['prev_expenses']) / (user_df['prev_expenses'] + 1e-6)
  
            try:
                train_num_cols = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
                fixed_categories = [col.replace('avg_exp_', '') for col in train_num_cols if col.startswith('avg_exp_')]
            except FileNotFoundError:
                fixed_categories = list(user_df['expenses_breakdown'].iloc[0].keys())
            for cat in fixed_categories:
                user_df[f'avg_exp_{cat}'] = user_df['expenses_breakdown'].apply(lambda d: d.get(cat, 0)).rolling(3).mean().fillna(0)
            return user_df

        df_agg = df_agg.groupby('user_id').apply(user_features, include_groups=False).reset_index()

        num_cols = [col for col in df_agg.columns if col.startswith('prev_') or col.startswith('trend_') or col.startswith('avg_')]
        try:
            scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        except FileNotFoundError:
            print("Error: Scaler file not found.")
            raise
        features = scaler.transform(df_agg[num_cols])

        try:
            train_num_cols = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
            if set(num_cols) != set(train_num_cols):
                missing_cols = [col for col in train_num_cols if col not in num_cols]
                if missing_cols:
                    for col in missing_cols:
                        df_agg[col] = 0
                    num_cols = train_num_cols
                    features = scaler.transform(df_agg[num_cols])

                extra_cols = [col for col in num_cols if col not in train_num_cols]
                if extra_cols:
                    print(f"Removing extra columns: {extra_cols}")
                    df_agg = df_agg.drop(columns=extra_cols)
                    num_cols = [col for col in num_cols if col not in extra_cols]
                    features = scaler.transform(df_agg[num_cols])
        except FileNotFoundError:
            print("Warning: feature_columns.pkl not found. Proceeding without feature alignment.")
        print(f"Shape of features after scaling: {features.shape}")

    if 'cluster' not in df_agg.columns:
        print("Warning: 'cluster' column not found in df_agg. Assigning default cluster 0 to all rows.")
        df_agg['cluster'] = 0

    required_cols = ['user_id', 'month', 'income', 'expenses', 'cluster']
    if not all(col in df_agg.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_agg.columns]
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

    df_agg = df_agg.sort_values(['user_id', 'month']).reset_index(drop=True)
    last_idx = df_agg.groupby('user_id').tail(1).index
    X_last = features[last_idx]
    last_month = df_agg.loc[last_idx].reset_index(drop=True)

    print("Loading models...")
    models_income = {}
    models_expenses = {}
    available_clusters = []

    for c in range(5):
        income_candidates = [
            os.path.join(models_dir, f"income_cluster_{c}.pkl"),
            os.path.join(models_dir, f"xgb_income_cluster_{c}.pkl"),
        ]
        expenses_candidates = [
            os.path.join(models_dir, f"expenses_cluster_{c}.pkl"),
            os.path.join(models_dir, f"xgb_expenses_cluster_{c}.pkl"),
        ]

        income_model = None
        expenses_model = None
        income_path_used = None
        expenses_path_used = None

        for p in income_candidates:
            if os.path.exists(p):
                income_model = joblib.load(p)
                income_path_used = p
                break

        for p in expenses_candidates:
            if os.path.exists(p):
                expenses_model = joblib.load(p)
                expenses_path_used = p
                break

        if income_model is not None and expenses_model is not None:
            models_income[c] = income_model
            models_expenses[c] = expenses_model
            print(f"Loaded models for cluster {c} from {income_path_used} / {expenses_path_used}")
            available_clusters.append(c)
        else:
            print(
                f"Warning: Models for cluster {c} not found at any expected path. Skipping cluster."
            )

    if not available_clusters:
        print("Error: No models found for any clusters. Cannot proceed with predictions.")
        raise FileNotFoundError("No valid model files found in ../models/.")


    def _scale_breakdown(breakdown, total_pred):
        """
        Scale a historical breakdown dict so that its values sum to total_pred.
        If breakdown is empty or sums to <= 0, put everything into 'Unknown'.
        """
        if not isinstance(breakdown, dict) or len(breakdown) == 0:
            return {"Unknown": float(total_pred)}

        hist_sum = sum(breakdown.values())
        if hist_sum <= 0:
            return {"Unknown": float(total_pred)}

        scaled = {cat: float(total_pred * val / hist_sum) for cat, val in breakdown.items()}
        return scaled

    results = []

    for i, row in last_month.iterrows():
        cluster_id = int(row['cluster'])
        if cluster_id not in models_income:
            continue

        x = X_last[i].reshape(1, -1)

        income_pred = float(models_income[cluster_id].predict(x)[0])
        expenses_pred = float(models_expenses[cluster_id].predict(x)[0])

        income_pred = max(0.0, income_pred)
        expenses_pred = max(0.0, expenses_pred)

        income_bd_hist = row.get('income_breakdown', {}) if isinstance(row.get('income_breakdown', {}), dict) else {}
        expenses_bd_hist = row.get('expenses_breakdown', {}) if isinstance(row.get('expenses_breakdown', {}), dict) else {}

        predicted_income_breakdown = _scale_breakdown(income_bd_hist, income_pred)
        predicted_expenses_breakdown = _scale_breakdown(expenses_bd_hist, expenses_pred)

        results.append(
            {
                "user_id": row["user_id"],
                "predicted_income": income_pred,
                "predicted_expenses": expenses_pred,
                "predicted_income_breakdown": predicted_income_breakdown,
                "predicted_expenses_breakdown": predicted_expenses_breakdown,
            }
        )

    if not results:
        raise RuntimeError("No predictions could be generated for any users.")

    predictions_df = pd.DataFrame(results)

    output_path = os.path.join(".", "result", "august_predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    return predictions_df
