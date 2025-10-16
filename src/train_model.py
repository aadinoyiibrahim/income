import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import optuna
import joblib
import os


def objective(trial, X_train, y_train, objective='reg:squarederror'):
    """
    Optuna objective for tuning XGBoost regressor hyperparameters using Bayesian optimization.
    """
    params = {
        'objective': objective,
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_models(df_agg, features, models_dir='../models'):
    """
    Train XGBoost regression models for income and expenses forecasting, per user cluster.

    Args:
        df_agg (pd.DataFrame): Preprocessed DataFrame with aggregated user-month data.
        features (np.ndarray): Preprocessed feature array.
        models_dir (str): Directory to save models and holdout data.

    Returns:
        tuple: (dict of income models, dict of expenses models)
    """
    # Ensure models directory exists with write permissions
    os.makedirs(models_dir, exist_ok=True)
    print(f"Created/verified models directory: {models_dir}")
    try:
        with open(os.path.join(models_dir, 'test_write.txt'), 'w') as f:
            f.write('test')
        os.remove(os.path.join(models_dir, 'test_write.txt'))
        print("Directory write permissions verified.")
    except PermissionError:
        print(f"Error: No write permissions for {models_dir}")
        raise

    # Validate input DataFrame
    required_cols = ['user_id', 'month', 'income', 'expenses', 'cluster']
    if not all(col in df_agg.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_agg.columns]
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

    # Debug: Check input data
    print("Columns in df_agg:", df_agg.columns)
    print(f"Shape of df_agg: {df_agg.shape}")
    print(f"Shape of features: {features.shape}")

    # Split users into train/holdout (~80/20)
    print("--1. Splitting users...")
    users = df_agg['user_id'].unique()
    train_users = np.random.choice(users, size=int(len(users) * 0.8), replace=False)
    train_df = df_agg[df_agg['user_id'].isin(train_users)]
    holdout_df = df_agg[~df_agg['user_id'].isin(train_users)]

    # Save holdout for evaluation
    print("--2. Saving holdout...")
    holdout_path = os.path.join(models_dir, 'holdout_df.pkl')
    joblib.dump(holdout_df, holdout_path)
    print(f"Saved holdout to: {holdout_path}")

    # For training, use Feb-Jun to predict next month
    print("--3. Filtering Feb-Jun...")
    # print(f'months: {train_df.columns}')
    print(f"months counts: {train_df['month'].value_counts()}")
    train_df = train_df[train_df['month'] < 7]  # Feb-Jun
    print(f"Training data shape after filtering: {train_df.shape}")

    # Create target variables: next month's income/expenses
    print("--4. Creating targets...")

    def create_targets(user_df):
        user_df = user_df.sort_values('month')
        user_df['target_income'] = user_df['income'].shift(-1)
        user_df['target_expenses'] = user_df['expenses'].shift(-1)
        return user_df.iloc[:-1]  # Drop last row (no target)

    train_df = train_df.groupby('user_id').apply(create_targets, include_groups=True).reset_index(drop=True)
    print(f"Training data shape after target creation: {train_df.shape}")

    # Use cluster labels from preprocessing
    print("--5. Training models per cluster...")
    models_income = {}
    models_expenses = {}
    default_model_trained = False
    for cluster in train_df['cluster'].unique():
        cluster_df = train_df[train_df['cluster'] == cluster]
        cluster_idx = train_df.index[train_df['cluster'] == cluster]
        X = features[cluster_idx]  # Align features with cluster
        y_inc = cluster_df['target_income']
        y_exp = cluster_df['target_expenses']

        # Skip if insufficient data
        if len(cluster_df) < 10 or y_inc.isna().all() or y_exp.isna().all():
            print(
                f"Skipping cluster {cluster}: insufficient data (rows: {len(cluster_df)}, income NaNs: {y_inc.isna().sum()}, expenses NaNs: {y_exp.isna().sum()})")
            continue

        try:
            # Tune and train for income
            print(f"Training income model for cluster {cluster}...")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X, y_inc).score(X, y_inc), n_trials=20)
            models_income[cluster] = objective(study.best_trial, X, y_inc)
            income_model_path = os.path.join(models_dir, f'xgb_income_cluster_{cluster}.pkl')
            joblib.dump(models_income[cluster], income_model_path)
            print(f"Saved income model for cluster {cluster} to {income_model_path}")

            # Tune and train for expenses
            print(f"Training expenses model for cluster {cluster}...")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, X, y_exp).score(X, y_exp), n_trials=20)
            models_expenses[cluster] = objective(study.best_trial, X, y_exp)
            expenses_model_path = os.path.join(models_dir, f'xgb_expenses_cluster_{c}.pkl')
            joblib.dump(models_expenses[cluster], expenses_model_path)
            print(f"Saved expenses model for cluster {cluster} to {expenses_model_path}")

            # Mark default model as trained if cluster 0 is processed
            if cluster == 0:
                default_model_trained = True
        except Exception as e:
            print(f"Error training or saving model for cluster {cluster}: {e}")
            continue

    # Train a default model if no models were trained
    if not models_income or not default_model_trained:
        print("No models trained or default model missing. Training default model for cluster 0...")
        try:
            X = features[train_df.index]
            y_inc = train_df['target_income']
            y_exp = train_df['target_expenses']
            if len(train_df) >= 10 and not y_inc.isna().all() and not y_exp.isna().all():
                study = optuna.create_study(direction='minimize')
                study.optimize(lambda trial: objective(trial, X, y_inc).score(X, y_inc), n_trials=20)
                models_income[0] = objective(study.best_trial, X, y_inc)
                income_model_path = os.path.join(models_dir, f'xgb_income_cluster_0.pkl')
                joblib.dump(models_income[0], income_model_path)
                print(f"Saved default income model for cluster 0 to {income_model_path}")

                study = optuna.create_study(direction='minimize')
                study.optimize(lambda trial: objective(trial, X, y_exp).score(X, y_exp), n_trials=20)
                models_expenses[0] = objective(study.best_trial, X, y_exp)
                expenses_model_path = os.path.join(models_dir, f'xgb_expenses_cluster_0.pkl')
                joblib.dump(models_expenses[0], expenses_model_path)
                print(f"Saved default expenses model for cluster 0 to {expenses_model_path}")
            else:
                print("Cannot train default model: insufficient data or all NaN targets.")
        except Exception as e:
            print(f"Error training or saving default model: {e}")
            raise RuntimeError("Failed to train or save any models.")

    print("Training complete. Trained models for clusters:", list(models_income.keys()))
    return models_income, models_expenses





    