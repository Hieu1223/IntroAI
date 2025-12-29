from model import Model
from dataset import HousingDataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_DIR = 'images/decision_tree'
os.makedirs(IMG_DIR, exist_ok=True)


class DecisionTree(Model):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=123)

    def train(self, dataset):
        self.model.fit(dataset.x, dataset.y)

    def evaluate(self, data):
        return self.model.predict(data.x)


def main():
    train_dataset = HousingDataset(mode="train")
    test_dataset = HousingDataset(mode="test")

    tree = DecisionTree(max_depth=6)
    tree.train(train_dataset)

    # Predictions
    y_train_pred = tree.evaluate(train_dataset)
    y_test_pred = tree.evaluate(test_dataset)

    print("Train MSE:", mse(y_train_pred, train_dataset.y))
    print("Test MSE:", mse(y_test_pred, test_dataset.y))

    # --------------------------
    # Prediction vs Truth Plot
    # --------------------------
    y_true_test = test_dataset.y * test_dataset.y_div_factor
    y_pred_test = y_test_pred * test_dataset.y_div_factor

    # Clip outliers (keep central 98%)
    low, high = np.percentile(y_true_test, [1, 99])
    mask = (y_true_test >= low) & (y_true_test <= high)

    plt.figure()
    plt.scatter(y_true_test[mask], y_pred_test[mask], alpha=0.5)
    plt.plot([low, high], [low, high], 'r')  # y=x reference line
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("Decision Tree: Prediction vs Ground Truth (98% clipped)")
    plt.savefig(os.path.join(IMG_DIR, "prediction_vs_truth.png"))
    plt.close()

    # --------------------------
    # Residual Plot
    # --------------------------
    residuals = y_test_pred - test_dataset.y
    plt.figure()
    plt.scatter(y_pred_test, residuals * test_dataset.y_div_factor, alpha=0.5)
    plt.axhline(0, color='r')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")
    plt.title("Decision Tree: Residual Plot")
    plt.savefig(os.path.join(IMG_DIR, "residuals.png"))
    plt.close()


if __name__ == "__main__":
    main()
