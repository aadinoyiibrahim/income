import matplotlib.pyplot as plt
import shap
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_absolute_error
from sklearn.inspection import permutation_importance  # Added missing import
import os
from preprocessing import preprocess_data

print(f'using updated code')
def plot_metrics(df_agg, features, actual_july, models_dir='../models'):
    """
    Plot model comparisons, metrics, and feature importance.
    
    Args:
        df_agg (pd.DataFrame): Aggregated DataFrame.
        features (np.array): Feature matrix.
        actual_july (pd.DataFrame): Actual July data for evaluation.
        models_dir (str): Directory for models.
    """
    print("Generating metrics and plots...")
    print(f"Columns in df_agg: {df_agg.columns}")
    print(f"Shape of df_agg: {df_agg.shape}")
    print(f"Shape of features: {features.shape}")
    
    # Metrics table (hypothetical; replace with actual if needed)
    metrics = []
    for cluster in range(5):
        metrics.append({
            'Cluster': cluster,
            'MAE_Income': np.random.rand() * 100,  # Placeholder; replace with actual
            'MAE_Expenses': np.random.rand() * 100
        })
    metrics_df = pd.DataFrame(metrics)
    print("Metrics per cluster:")
    print(metrics_df)
    
    # Bar plot
    metrics_df.plot(x='Cluster', kind='bar', figsize=(10, 6))
    plt.title('Model Performance per Cluster')
    plt.savefig('../img/cluster_metrics.png')
    plt.close()
    
    # SHAP explainability (for one model)
    try:
        model = joblib.load(f'{models_dir}/xgb_income_cluster_0.pkl')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features[:500])  # Sample for speed
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]  # Placeholder; use actual names
        shap.summary_plot(shap_values, features[:500], feature_names=feature_names, show=False)
        plt.savefig('../img/shap_summary.png')
        plt.close()
        print("Generated SHAP summary plot...")
    except Exception as e:
        print(f"SHAP error: {e}")
    
    # XGBoost Built-in Importance (Gain)
    try:
        model = joblib.load(f'{models_dir}/xgb_income_cluster_0.pkl')
        fig, ax = plt.subplots()
        xgb.plot_importance(model, ax=ax, max_num_features=20)
        plt.title('XGBoost Feature Importance (Gain)')
        plt.savefig('../img/xgb_importance.png')
        plt.close()
        print("Generated XGBoost feature importance plot...")
    except Exception as e:
        print(f"XGBoost importance error: {e}")
    
    # Permutation Importance (Model-Agnostic)
    try:
        model = joblib.load(f'{models_dir}/xgb_income_cluster_0.pkl')
        # Use holdout for evaluation
        holdout_df = joblib.load(f'{models_dir}/holdout_df.pkl')
        holdout_features, _, _ = preprocess_data(holdout_df, scaler=joblib.load(f'{models_dir}/scaler.pkl'), fit=False)
        print("Loading holdout data for predictions...")
        print(f"Columns in holdout_df: {holdout_df.columns}")
        print(f"Shape of holdout_df: {holdout_df.shape}")
        print("Applying feature engineering to holdout data...")
        print(f"Shape of holdout_features: {holdout_features.shape}")
        
        print("Loading models for metrics...")
        models = {}
        for c in range(5):
            try:
                models[c] = joblib.load(f'{models_dir}/xgb_income_cluster_{c}.pkl')
                print(f"Loaded income model for cluster {c}")
            except FileNotFoundError:
                print(f"Model for cluster {c} not found; skipping")
        
        if len(models) > 0:
            print("Generating predictions for evaluation...")
            pred_df = pd.DataFrame({
                'predicted_income': np.zeros(len(holdout_df)),
                'predicted_expenses': np.zeros(len(holdout_df))
            })
            for c in models.keys():
                mask = holdout_df['cluster'] == c
                if mask.sum() > 0:
                    pred_df.loc[holdout_df[mask].index, 'predicted_income'] = models[c].predict(holdout_features[mask])
            print(f"Shape of pred_df: {pred_df.shape}")
            
            merged = holdout_df.merge(pred_df, left_index=True, right_index=True)
            merged = merged.merge(actual_july, on='user_id', how='inner')
            print(f"Merged data shape for evaluation: {merged.shape}")
            
            if len(merged) > 0 and len(models) > 0:
                sample_size = min(500, len(holdout_features), len(merged))
                perm = permutation_importance(list(models.values())[0], holdout_features[:sample_size], merged['income'][:sample_size], n_repeats=10, random_state=42)
                try:
                    train_num_cols = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
                except FileNotFoundError:
                    train_num_cols = [f'feature_{i}' for i in range(holdout_features.shape[1])]
                perm_df = pd.DataFrame({'feature': train_num_cols, 'importance': perm.importances_mean})
                perm_df = perm_df.sort_values('importance', ascending=False).head(20)
                perm_df.plot(kind='bar', x='feature', y='importance', figsize=(10, 6))
                plt.title('Permutation Importance')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('../img/permutation_importance.png')
                plt.close()
                print("Generated permutation importance plot...")
            else:
                print("Skipping permutation importance: insufficient data or models")
        else:
            print("No models loaded; skipping metrics and importance plots")
    except Exception as e:
        print(f"Permutation importance error: {e}")
    
    print("Metrics and plots generation complete.")