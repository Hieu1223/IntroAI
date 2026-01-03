from model import Model
from ames_dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from plot_gen import *

class MLP(Model):
    def __init__(self, layers=(13, 16, 1)):
        self.layers = layers
        self.n_layers = len(layers) - 1
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            w = np.random.randn(layers[i], layers[i + 1]) * 0.01
            b = np.zeros(layers[i + 1])
            self.weights.append(w)
            self.biases.append(b)

        self.loss_history = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        for i in range(self.n_layers - 1):
            Z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            self.activations.append(A)
        Z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.activations.append(Z_out)
        return Z_out

    def train(self, dataset, steps=1000, lr=0.01):
        X = dataset.x
        y = dataset.y.reshape(-1, 1)
        n = len(X)

        for step in range(steps):
            A = [X]
            for i in range(self.n_layers - 1):
                Z = np.dot(A[-1], self.weights[i]) + self.biases[i]
                A.append(self.relu(Z))
            Z_out = np.dot(A[-1], self.weights[-1]) + self.biases[-1]
            A.append(Z_out)
            y_hat = Z_out

            loss = np.mean((y_hat - y) ** 2)
            self.loss_history.append(loss)

            dA = 2 * (y_hat - y) / n
            for i in reversed(range(self.n_layers)):
                A_prev = A[i]
                W = self.weights[i]

                dW = np.dot(A_prev.T, dA)
                db = np.sum(dA, axis=0)

                if i > 0:
                    dA = np.dot(dA, W.T) * self.relu_grad(A_prev)

                self.weights[i] -= lr * dW
                self.biases[i] -= lr * db

            if step % 50 == 0:
                print(f"Step {step}, MSE: {loss:.4f}")
            if step == steps-1:
                print(f"Step {step}, MSE: {loss:.4f}")
                return

    def evaluate(self, dataset):
        return self.forward(dataset.x)


def main():
    np.random.seed(123)
    IMG_DIR = "images/neural_network"
    os.makedirs(IMG_DIR, exist_ok=True)

    train_dataset = AmesHousingDataset(mode="train")
    test_dataset = AmesHousingDataset(mode="test")

    model = MLP(layers=(train_dataset.n_features, 100, 1))
    model.train(train_dataset, steps=1000, lr=0.05)


    y_pred = model.evaluate(test_dataset)
    print("Test MSE:", mean_squared_error(y_pred, test_dataset.y))


    y_pred_flat = y_pred.flatten()
    y_true_flat = (test_dataset.y * test_dataset.y_div_factor).flatten()
    y_pred_plot = y_pred_flat * test_dataset.y_div_factor


    plot_loss_curve(
        loss_history=model.loss_history,
        img_path=os.path.join(IMG_DIR, "loss_curve.png"),
        title="MLP Training Loss Curve"
    )


    plot_prediction_vs_truth(
        y_true=y_true_flat,
        y_pred=y_pred_plot,
        img_path=os.path.join(IMG_DIR, "prediction_vs_truth.png"),
        clip_percentiles=(1,99),
        title="MLP: Prediction vs Ground Truth"
    )

    plot_residuals(
        y_true=y_true_flat,
        y_pred=y_pred_plot,
        img_path=os.path.join(IMG_DIR, "residuals.png"),
        title="MLP: Residual Plot"
    )

if __name__ == "__main__":
    main()
