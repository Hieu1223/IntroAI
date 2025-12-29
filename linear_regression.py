from model import Model
from datasets import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


class LinearRegression(Model):
    def __init__(self, n_features):
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        self.loss_history = []

    def train(self, dataset, steps=1000, lr=0.01):
        X, y = dataset.x, dataset.y
        n = len(X)

        for step in range(steps):
            y_hat = X @ self.w + self.b

            dw = (X.T @ (y_hat - y)) / n
            db = np.mean(y_hat - y)

            self.w -= lr * dw
            self.b -= lr * db

            loss = np.mean((y_hat - y) ** 2)
            self.loss_history.append(loss)

            if step % 50 == 0:
                print(f"Step {step}, MSE: {loss:.4f}")

    def evaluate(self, data):
        return data.x @ self.w + self.b




def main():
    np.random.seed(123)
    IMG_DIR = 'images/linear_regression'
    train_dataset = AmesHousingDataset(mode='train')
    test_dataset = AmesHousingDataset(mode='test')

    model = LinearRegression(n_features=train_dataset.n_features)
    model.train(train_dataset, steps=100, lr=0.05)

    y_pred = model.evaluate(test_dataset)
    print("Test MSE:", mean_squared_error(y_pred, test_dataset.y))

    # --------------------------
    # Loss Curve
    # --------------------------
    plt.figure()
    plt.plot(model.loss_history)
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.savefig(IMG_DIR + '/' +"loss_curve.png")
    plt.close()

    # --------------------------
    # Prediction vs Truth
    # --------------------------
    plt.figure()
    y_true = test_dataset.y * test_dataset.y_div_factor
    y_pred_plot = y_pred * test_dataset.y_div_factor
    low, high = np.percentile(y_true, [1, 99])
    mask = (y_true >= low) & (y_true <= high)

    plt.scatter(y_true[mask], y_pred_plot[mask], alpha=0.5)
    plt.plot([low, high], [low, high], 'r--')  # y = x reference line
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("Prediction vs Ground Truth (98% clipped)")
    plt.savefig(IMG_DIR + "/prediction_vs_truth.png")
    plt.close()

    # --------------------------
    # Residual Plot
    # --------------------------
    residuals = y_pred - test_dataset.y
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")
    plt.title("Residual Plot")
    plt.savefig(IMG_DIR + '/' +"residuals.png")
    plt.close()

    # --------------------------
    # Feature Coefficients
    # --------------------------
    coefs = model.w.flatten()  # ensure it's 1D

    # Get indices of 5 largest absolute coefficients
    top5_idx = np.argsort(np.abs(coefs))[-5:][::-1]  # descending order

    # Get corresponding feature names
    if hasattr(train_dataset, 'feature_names'):
        feature_names = train_dataset.feature_names[top5_idx]
    else:
        feature_names = [f"Feature {i}" for i in top5_idx]

    # Plot
    plt.figure(figsize=(10,6))
    plt.bar(range(5), coefs[top5_idx])
    plt.xticks(range(5), feature_names, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.title("Top 5 Most Significant Linear Regression Coefficients")
    plt.tight_layout()
    plt.savefig(IMG_DIR + "/feature_coefficients.png")
    plt.close()



if __name__ == "__main__":
    main()