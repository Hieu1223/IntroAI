import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dataset import HousingDataset  # your dataset class

def main():
    np.random.seed(123)
    IMG_DIR = "images/sklearn_regression"
    os.makedirs(IMG_DIR, exist_ok=True)

    # Load datasets
    train_dataset = HousingDataset(mode="train")
    test_dataset = HousingDataset(mode="test")

    # Train linear regression using sklearn
    model = LinearRegression()
    model.fit(train_dataset.x, train_dataset.y)

    # Predict
    y_train_pred = model.predict(train_dataset.x)
    y_test_pred = model.predict(test_dataset.x)

    # Compute MSE
    print("Train MSE:", mean_squared_error(train_dataset.y, y_train_pred))
    print("Test MSE:", mean_squared_error(test_dataset.y, y_test_pred))

    # -------------------------
    # 1️⃣ Loss curve (not available in sklearn, use residuals on train)
    residuals_train = y_train_pred - train_dataset.y
    plt.figure()
    plt.plot(np.arange(len(residuals_train)), residuals_train ** 2)
    plt.xlabel("Sample")
    plt.ylabel("Squared Residual")
    plt.title("Training Squared Residuals")
    plt.savefig(os.path.join(IMG_DIR, "train_residuals.png"))
    plt.close()

    # -------------------------
    # 2️⃣ Prediction vs Ground Truth
    y_true = test_dataset.y * test_dataset.y_div_factor
    y_pred_plot = y_test_pred * test_dataset.y_div_factor

    low, high = np.percentile(y_true, [1, 99])
    mask = (y_true >= low) & (y_true <= high)

    plt.figure()
    plt.scatter(y_true[mask], y_pred_plot[mask], alpha=0.5)
    plt.plot([low, high], [low, high], color='r')
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title("Prediction vs Ground Truth (98% clipped)")
    plt.savefig(os.path.join(IMG_DIR, "prediction_vs_truth.png"))
    plt.close()

    # -------------------------
    # 3️⃣ Residuals
    residuals = (y_test_pred - test_dataset.y) * test_dataset.y_div_factor
    plt.figure()
    plt.scatter(y_pred_plot, residuals, alpha=0.5)
    plt.axhline(0, color='r')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")
    plt.title("Residual Plot")
    plt.savefig(os.path.join(IMG_DIR, "residuals.png"))
    plt.close()

if __name__ == "__main__":
    main()
