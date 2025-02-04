import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import expit as sigmoid


class LogisticRegressionModels:
    def __init__(
        self,
        model_type="baseline",
        learning_rate=0.01,
        max_iter=1000,
        noise_rate=0.01,
        weight_positive=1.0,
        weight_negative=1.0,
        alpha=0.25,
        gamma=2.0,
    ):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.noise_rate = noise_rate
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.weights = None
        self.alpha = alpha
        self.gamma = gamma
        self.bias = 0
        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        """
        Compute the cross-entropy loss for
        the given data and labels.
        Param:
            X: np.ndarray, input data
            y: np.ndarray, labels
        Return:
            float, cross-entropy loss
        """
        logits = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(logits)
        small_number = 1e-10  # to avoid log(0)

        if self.model_type == "baseline":
            loss_output = -np.mean(
                y * np.log(predictions + small_number)
                + (1 - y) * np.log(1 - predictions + small_number)
            )
            return loss_output

        elif self.model_type == "cost_sensitive":
            first_term = self.weight_positive * y * np.log(predictions + small_number)
            second_term = (
                self.weight_negative * (1 - y) * np.log(1 - predictions + small_number)
            )
            loss_output = -np.mean(first_term + second_term)
            return loss_output
        elif self.model_type == "focal":
            loss_output = -np.mean(
                self.alpha
                * ((1 - predictions) ** self.gamma)
                * y
                * np.log(predictions + small_number)
                + (1 - self.alpha)
                * (predictions**self.gamma)
                * (1 - y)
                * np.log(1 - predictions + small_number)
            )
            return loss_output

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            logits = np.dot(X_train, self.weights) + self.bias
            predictions = self.sigmoid(logits)

            if self.model_type == "cost_sensitive":
                weights = (
                    y_train * self.weight_positive
                    + (1 - y_train) * self.weight_negative
                )
                dw = np.dot(X_train.T, weights * (predictions - y_train)) / n_samples
                db = np.sum(weights * (predictions - y_train)) / n_samples
            elif self.model_type == "focal":
                focal_weights = self.alpha * (
                    (1 - predictions) ** self.gamma
                ) * y_train + (1 - self.alpha) * (predictions**self.gamma) * (
                    1 - y_train
                )
                dw = (
                    np.dot(X_train.T, focal_weights * (predictions - y_train))
                    / n_samples
                )
                db = np.sum(focal_weights * (predictions - y_train)) / n_samples
            else:
                dw = np.dot(X_train.T, (predictions - y_train)) / n_samples
                db = np.sum(predictions - y_train) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            train_loss = self.compute_loss(X_train, y_train)
            val_loss = self.compute_loss(X_val, y_val)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

    def predict_proba(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self.sigmoid(logits)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)

    def export_results(self):
        """results only"""
        result = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "weights": self.weights,
            "bias": self.bias,
        }
        return result

    def print_results(self, X_train, y_train, X_val, y_val, X_test, y_test):
        train_preds = self.predict_proba(X_train)
        val_preds = self.predict_proba(X_val)
        test_preds = self.predict_proba(X_test)

        train_accuracy = accuracy_score(y_train, self.predict(X_train))
        val_accuracy = accuracy_score(y_val, self.predict(X_val))
        test_accuracy = accuracy_score(y_test, self.predict(X_test))

        train_loss = log_loss(y_train, train_preds)
        val_loss = log_loss(y_val, val_preds)
        test_loss = log_loss(y_test, test_preds)

        print(f"  Training Accuracy: {train_accuracy:.4f}, Log Loss: {train_loss:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}, Log Loss: {val_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}, Log Loss: {test_loss:.4f}\n")
