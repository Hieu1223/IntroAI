from model import Model
from datasets import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from plot_gen import *

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
    os.makedirs(IMG_DIR, exist_ok=True)

    # Load datasets
    train_dataset = AmesHousingDataset(mode='train')
    test_dataset = AmesHousingDataset(mode='test')

    # Train Linear Regression
    model = LinearRegression(n_features=train_dataset.n_features)
    model.train(train_dataset, steps=100, lr=0.05)

    # Predictions
    y_pred = model.evaluate(test_dataset)
    print("Test MSE:", mean_squared_error(y_pred, test_dataset.y))

    # Scale back to original prices
    y_true = test_dataset.y * test_dataset.y_div_factor
    y_pred_plot = y_pred * test_dataset.y_div_factor

    # --------------------------
    # Plots using modular functions
    # --------------------------
    # Loss Curve
    plot_loss_curve(
        loss_history=model.loss_history,
        img_path=os.path.join(IMG_DIR, "loss_curve.png"),
        title="Linear Regression Training Loss Curve"
    )

    # Prediction vs Truth
    plot_prediction_vs_truth(
        y_true=y_true,
        y_pred=y_pred_plot,
        img_path=os.path.join(IMG_DIR, "prediction_vs_truth.png"),
        clip_percentiles=(1,99),
        title="Linear Regression: Prediction vs Ground Truth"
    )

    # Residual Plot
    residuals = (y_pred - test_dataset.y) * test_dataset.y_div_factor
    plot_residuals(
        y_true=y_true,
        y_pred=y_pred_plot,
        img_path=os.path.join(IMG_DIR, "residuals.png"),
        title="Linear Regression: Residual Plot"
    )

    plot_top_linear_coefficients(
        model=model,
        train_dataset=train_dataset,
        img_path=os.path.join(IMG_DIR, "feature_coefficients.png"),
        top_n=5,
        title="Top 5 Most Significant Linear Regression Coefficients"
    )





if __name__ == "__main__":
    main()