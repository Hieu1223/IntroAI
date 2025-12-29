from model import Model
from dataset import HousingDataset
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
        self.model.fit(dataset.x, dataset.y)

    def evaluate(self, dataset):
        return self.model.predict(dataset.x)


def main():
    train_dataset = HousingDataset(mode="train")
    test_dataset = HousingDataset(mode="test")

    rf = RandomForest(n_estimators=200, max_depth=20, seed=123)
    rf.train(train_dataset)

    # Predictions
    y_train_pred = rf.evaluate(train_dataset)
    y_test_pred = rf.evaluate(test_dataset)

    print("Train MSE:", mse(y_train_pred, train_dataset.y))
    print("Test MSE:", mse(y_test_pred, test_dataset.y))


    y_true_test = test_dataset.y * test_dataset.y_div_factor
    y_pred_test = y_test_pred * test_dataset.y_div_factor

    low, high = np.percentile(y_true_test, [1, 99])
    mask = (y_true_test >= low) & (y_true_test <= high)

    plt.figure()
    plt.scatter(y_true_test[mask], y_pred_test[mask], alpha=0.5)
    plt.plot([low, high], [low, high], 'r--')  # y=x reference line
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("Random Forest: Prediction vs Ground Truth (98% clipped)")
    plt.savefig(os.path.join(IMG_DIR, "prediction_vs_truth.png"))
    plt.close()

    residuals = y_test_pred - test_dataset.y
    plt.figure()
    plt.scatter(y_pred_test, residuals * test_dataset.y_div_factor, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")
    plt.title("Random Forest: Residual Plot")
    plt.savefig(os.path.join(IMG_DIR, "residuals.png"))
    plt.close()


if __name__ == "__main__":
    main()
