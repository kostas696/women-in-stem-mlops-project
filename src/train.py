import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODELS = ["randomforest", "xgboost", "catboost"]

def load_data():
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    return X, y

def get_model(model_name):
    if model_name == "randomforest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "xgboost":
        return XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    elif model_name == "catboost":
        return CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

def save_model(model, model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    return str(model_path)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Women-in-STEM")

    for model_name in MODELS:
        with mlflow.start_run(run_name=f"{model_name}_run"):
            model = get_model(model_name)
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_param("model", model_name)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, name=f"{model_name}_artifact")
            path = save_model(model, model_name)

            print(f"{model_name} training complete. Metrics:", metrics)
            print(f"Model saved to {path}")

if __name__ == "__main__":
    main()