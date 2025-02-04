import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import plotter
import numpy as np
import sklearn

from preprocessing import load_and_preprocess
from LRmodel import LogisticRegressionModels
import appendix

data_dir = "./data/2016-09-19_79351_training_feb_june.csv"

X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoders = load_and_preprocess(
    data_dir, training=True
)

# hyperparameters
n_estimator = 100
seedS = 42
lr = 0.01
max_iter = 1000
max_depth = 15
class_weight = {0: 1.45, 1: 1.0}

"""
-------------------------
Define models
-------------------------
"""
models = {
    "Baseline RF": RandomForestClassifier(n_estimators=n_estimator, random_state=seedS),
    "Weighted RF": RandomForestClassifier(
        n_estimators=n_estimator,
        class_weight=class_weight,
        max_depth=10,
        random_state=seedS,
    ),
    "Baseline LR": LogisticRegressionModels(
        model_type="baseline", learning_rate=lr, max_iter=max_iter
    ),
    "Cost-sensitive LR": LogisticRegressionModels(
        model_type="cost_sensitive",
        weight_positive=class_weight[1],
        weight_negative=class_weight[0],
        learning_rate=lr,
        max_iter=max_iter,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=max_iter,
        verbose=False,
        random_state=seedS,
    ),
}

# Train, evaluate, and save models
results = {}
for name, model in models.items():
    print(f"Training {name}")

    if isinstance(model, (RandomForestClassifier, MLPClassifier)):
        model.fit(X_train, y_train)
    elif isinstance(model, LogisticRegressionModels):
        model.fit(X_train, y_train, X_val, y_val)
    else:
        raise ValueError("Model not supported")

    show_curves = False
    if show_curves:
        if isinstance(model, (LogisticRegressionModels)):
            plt.figure(figsize=(10, 6))
            plt.plot(model.train_losses, label="Training Loss")
            plt.plot(model.val_losses, label="Validation Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.show()

    # Preds
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    if isinstance(model, (RandomForestClassifier, MLPClassifier)):
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba_test)
    elif isinstance(model, LogisticRegressionModels):
        y_pred_proba_test = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred_proba_test)
    else:
        raise ValueError("Model not supported")

    # Save model
    joblib.dump(model, f"models/{name.replace(' ', '_')}_model.pkl")

    results[name] = {
        "Train Accuracy": train_accuracy,
        "Val Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
        "F1 Score": f1,
        "AUC": auc,
        "Model": model,
        "Model Path": f"models/{name.replace(' ', '_')}_model.pkl",
    }

    appendix.save_results(results, filename="results_storage.pkl")

    print(f"--- {name} saved ---")
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Val Accuracy: {val_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

plot_accuracy = True
if plot_accuracy:  # show accurary plot
    plotter.plot_barplot(
        indices=np.arange(len(results)),
        train_accuracies=[results[model]["Train Accuracy"] for model in results],
        val_accuracies=[results[model]["Val Accuracy"] for model in results],
        test_accuracies=[results[model]["Test Accuracy"] for model in results],
        labels=list(results.keys()),
        savefig="./result/accuracy.png",
    )

plot_auc = True
if plot_auc:  # show auc plot & save
    plotter.plot_auc_curves(
        results,
        X_test,
        y_test,
        savefig="./result/auc.png",
    )

# Save preprocessing artifacts (Scaler & Encoders)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")
