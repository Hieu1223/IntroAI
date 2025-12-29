from model import Model
from datasets import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_DIR = 'images/random_forest'
os.makedirs(IMG_DIR, exist_ok=True)

class RandomForest(Model):
    def __init__(self, n_estimators=100, max_depth=None, seed=123):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed
        )

    def train(self, dataset):
        # Convert dict to list in correct order
        if hasattr(dataset, "feature_names"):
            self.feature_names = [dataset.feature_names[i] for i in range(len(dataset.feature_names))]
        else:
            self.feature_names = None
        self.model.fit(dataset.x, dataset.y)


    def evaluate(self, dataset):
        return self.model.predict(dataset.x)

    def plot_feature_importance(self):
        if self.feature_names is None:
            print("No feature names provided.")
            return

        importances = self.model.feature_importances_
        # Indices of top 5 features
        indices = np.argsort(importances)[::-1][:5]
        names = [self.feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10,6))
        plt.bar(range(len(top_importances)), top_importances)
        plt.xticks(range(len(top_importances)), names, rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Decision Tree Feature Importance (Top 5)")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"))
        plt.close()


def main():
    train_dataset = AmesHousingDataset(mode="train",normalize=False,onehot=False)
    test_dataset = AmesHousingDataset(mode="test",normalize=False,onehot=False)

    rf = RandomForest(n_estimators=200, max_depth=7, seed=123)
    rf.train(train_dataset)

    # Predictions
    y_train_pred = rf.evaluate(train_dataset)
    y_test_pred = rf.evaluate(test_dataset)

    print("Train MSE:", mse(y_train_pred, train_dataset.y))
    print("Test MSE:", mse(y_test_pred, test_dataset.y))

    # --------------------------
    # Prediction vs Truth Plot
    # --------------------------
    y_true_test = test_dataset.y * test_dataset.y_div_factor
    y_pred_test = y_test_pred * test_dataset.y_div_factor

    low, high = np.percentile(y_true_test, [1, 99])
    mask = (y_true_test >= low) & (y_true_test <= high)

    plt.figure()
    plt.scatter(y_true_test[mask], y_pred_test[mask], alpha=0.5)
    plt.plot([low, high], [low, high], 'r--')
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("Random Forest: Prediction vs Ground Truth (98% clipped)")
    plt.savefig(os.path.join(IMG_DIR, "prediction_vs_truth.png"))
    plt.close()

    # --------------------------
    # Residual Plot
    # --------------------------
    residuals = y_test_pred - test_dataset.y
    plt.figure()
    plt.scatter(y_pred_test, residuals * test_dataset.y_div_factor, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")
    plt.title("Random Forest: Residual Plot")
    plt.savefig(os.path.join(IMG_DIR, "residuals.png"))
    plt.close()

    # --------------------------
    # Feature Importance Plot
    # --------------------------
    rf.plot_feature_importance()


if __name__ == "__main__":
    main()
