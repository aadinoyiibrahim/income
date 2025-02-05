"""
Project: case study

this file plots the results

Author: Abdullahi A. Ibrahim
date: 05-02-2025
"""

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os


def plot_auc_curves(results, X_test, y_test, figsize=(10, 6), savefig=None):
    """
    ROC curves here.
    """
    plt.figure(figsize=figsize)
    for name, metrics in results.items():
        model = metrics["Model"]
        if model.__class__.__name__ in ["RandomForestClassifier", "MLPClassifier"]:
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        elif model.__class__.__name__ == "LogisticRegressionModels":
            y_pred_proba_test = model.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        else:
            continue

        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["AUC"]:.2f})')
    plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best", fontsize=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if savefig is not None:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)  # Ensure directory exists
        plt.savefig(savefig, bbox_inches="tight")
        print(f"Figure saved to {savefig}")
    else:
        plt.show()


def plot_barplot(
    indices,
    train_accuracies,
    val_accuracies,
    test_accuracies,
    labels,
    bar_width=0.2,
    figsize=(16, 8),
    savefig=None,
):

    plt.figure(figsize=figsize)
    plt.bar(indices, train_accuracies, width=bar_width, label="training", alpha=0.8)
    plt.bar(
        indices + bar_width,
        val_accuracies,
        width=bar_width,
        label="validation",
        alpha=0.8,
    )
    plt.bar(
        indices + 2 * bar_width,
        test_accuracies,
        width=bar_width,
        label="test",
        alpha=0.8,
    )

    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(indices + bar_width, labels, rotation=0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if savefig is not None:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, bbox_inches="tight")
        print(f"Figure saved to {savefig}")
    else:
        plt.show()
