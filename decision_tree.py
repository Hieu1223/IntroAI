from model import Model
from ames_dataset import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import os
from plot_gen import *

IMG_DIR = 'images/decision_tree'
os.makedirs(IMG_DIR, exist_ok=True)


class DecisionTree(Model):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=123)
        self.feature_names = None

    def train(self, dataset):
        if hasattr(dataset, "feature_names"):
            self.feature_names = dataset.feature_names
        self.model.fit(dataset.x, dataset.y)

    def evaluate(self, dataset):
        return self.model.predict(dataset.x)


def main():
    # Load datasets
    train_dataset = AmesHousingDataset(mode="train")
    test_dataset = AmesHousingDataset(mode="test")

    # Train Decision Tree
    tree = DecisionTree(max_depth=7)
    tree.train(train_dataset)

    # Predictions
    y_train_pred = tree.evaluate(train_dataset)
    y_test_pred = tree.evaluate(test_dataset)

    # Print MSE
    print("Train MSE:", mse(y_train_pred, train_dataset.y))
    print("Test MSE:", mse(y_test_pred, test_dataset.y))

    # Scale predictions and true values back to original
    y_true_test = test_dataset.y * test_dataset.y_div_factor
    y_pred_test = y_test_pred * test_dataset.y_div_factor

    # --------------------------
    # Use plotting functions
    # --------------------------
    plot_prediction_vs_truth(
        y_true=y_true_test,
        y_pred=y_pred_test,
        img_path=os.path.join(IMG_DIR, "prediction_vs_truth.png"),
        title="Decision Tree: Prediction vs Ground Truth"
    )

    plot_residuals(
        y_true=y_true_test,
        y_pred=y_pred_test,
        img_path=os.path.join(IMG_DIR, "residuals.png"),
        title="Decision Tree: Residual Plot"
    )

    plot_feature_importance(
        model=tree.model,
        feature_names=train_dataset.feature_names,
        img_path=os.path.join(IMG_DIR, "feature_importance.png"),
        top_n=5,
        title="Decision Tree Feature Importance"
    )



if __name__ == "__main__":
    main()
