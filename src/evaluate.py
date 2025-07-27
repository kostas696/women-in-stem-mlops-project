import numpy as np
import joblib
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"

MODELS = ["randomforest", "xgboost", "catboost"]

def load_test_data(test_size=0.2, random_state=42):
    X = np.load(DATA_PATH / "X.npy")
    y = np.load(DATA_PATH / "y.npy")
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

def main():
    X_test, y_test = load_test_data()

    print(f"{'Model':<15} | {'RMSE':<8} | {'MAE':<8} | {'R^2':<8}")
    print("-" * 50)

    mlflow.set_experiment("Women-in-STEM")

    for model_name in MODELS:
        model_file = MODEL_PATH / f"{model_name}.pkl"

        if not model_file.exists():
            print(f"Model {model_name} not found at {model_file}")
            continue

        model = joblib.load(model_file)
        metrics = evaluate_model(model, X_test, y_test)

        print(f"{model_name:<15} | {metrics['rmse']:<8.3f} | {metrics['mae']:<8.3f} | {metrics['r2']:<8.3f}")

        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            mlflow.log_params({"model": model_name})
            mlflow.log_metrics(metrics)

if __name__ == "__main__":
    main()