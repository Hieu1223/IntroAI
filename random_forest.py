from model import Model
from ames_dataset import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import os
from plot_gen import *
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
        if hasattr(dataset, "feature_names"):
            self.feature_names = [dataset.feature_names[i] for i in range(len(dataset.feature_names))]
        else:
            self.feature_names = None
        self.model.fit(dataset.x, dataset.y)


    def evaluate(self, dataset):
        return self.model.predict(dataset.x)



def main():
    train_dataset = AmesHousingDataset(mode="train")
    test_dataset = AmesHousingDataset(mode="test")

    rf = RandomForest(n_estimators=200, max_depth=10, seed=123)
    rf.train(train_dataset)
    
    y_train_pred = rf.evaluate(train_dataset)
    y_test_pred = rf.evaluate(test_dataset)
    print("Train MSE:", mse(y_train_pred, train_dataset.y))
    print("Test MSE:", mse(y_test_pred, test_dataset.y))

    y_true_test = test_dataset.y * test_dataset.y_div_factor
    y_pred_test = y_test_pred * test_dataset.y_div_factor

    plot_prediction_vs_truth(
        y_true_test, y_pred_test, 
        os.path.join(IMG_DIR, "prediction_vs_truth.png"),
        title="Random Forest: Prediction vs Ground Truth"
    )

    plot_residuals(
        y_true_test, y_pred_test, 
        os.path.join(IMG_DIR, "residuals.png"),
        title="Random Forest: Residual Plot"
    )

    plot_feature_importance(
        rf.model, train_dataset.feature_names, 
        os.path.join(IMG_DIR, "feature_importance.png"),
        top_n=5, title="Random Forest Feature Importance")
    
    print_top_feature_importance(
        rf.model, train_dataset.feature_names
    )


if __name__ == "__main__":
    main()
