import os
import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_vs_truth(y_true, y_pred, img_path, clip_percentiles=(1,99), title="Prediction vs Ground Truth"):
    low, high = np.percentile(y_true, clip_percentiles)
    mask = (y_true >= low) & (y_true <= high)
    
    plt.figure(figsize=(8,6))
    plt.scatter(y_true[mask], y_pred[mask], alpha=0.5)
    plt.plot([low, high], [low, high], 'r--')
    plt.xlabel("True Price")
    plt.ylabel("Predicted Price")
    plt.title(title + f" ({clip_percentiles[0]}-{clip_percentiles[1]}% clipped)")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def plot_residuals(y_true, y_pred, img_path, title="Residual Plot"):
    residuals = y_pred - y_true
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual Error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def print_top_feature_importance(model, feature_names,top_n = 5):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    names = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    print("Top feature importances:")
    for i, (name, importance) in enumerate(zip(names, top_importances), start=1):
        print(f"{i}. Feature: {name}, Importance: {importance:.6f}")

def plot_feature_importance(model, feature_names, img_path, top_n=5, title="Feature Importance"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10,6))
    plt.bar(range(top_n), top_importances)
    plt.xticks(range(top_n), names, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(title + f" (Top {top_n})")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def plot_loss_curve(loss_history, img_path, title="Training Loss Curve"):
    plt.figure(figsize=(8,6))
    plt.plot(loss_history)
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def plot_top_linear_coefficients(model, train_dataset, img_path, top_n=5, title="Top Linear Regression Coefficients"):
    coefs = model.w.flatten()

    top_idx = np.argsort(np.abs(coefs))[-top_n:][::-1]

    # Feature names
    if hasattr(train_dataset, 'feature_names'):
        feature_names = train_dataset.feature_names[top_idx]
    else:
        feature_names = [f"Feature {i}" for i in top_idx]

    top_coefs = coefs[top_idx]

    # Plot
    plt.figure(figsize=(10,6))
    plt.bar(range(top_n), top_coefs)
    plt.xticks(range(top_n), feature_names, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.title(title + f" (Top {top_n})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    plt.close()