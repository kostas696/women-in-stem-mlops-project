from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List
import os
from pathlib import Path

app = FastAPI(title="Women in STEM Model API")

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
DEFAULT_MODEL_PATH = MODEL_DIR / "randomforest.pkl"  # fallback

# Load artifacts
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Models dictionary
models = {
    "randomforest": joblib.load(MODEL_DIR / "randomforest.pkl"),
    "xgboost": joblib.load(MODEL_DIR / "xgboost.pkl"),
    "catboost": joblib.load(MODEL_DIR / "catboost.pkl"),
}


# Pydantic input model
class InputData(BaseModel):
    year: int
    female_enrollment: float
    gender_gap_index: float
    country: str
    stem_fields: str
    model_name: str = "randomforest"


@app.get("/")
def root():
    return {"message": "Welcome to the Women-in-STEM Predictor API!"}


@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert to DataFrame
        import pandas as pd
        input_df = pd.DataFrame([data.dict(exclude={"model_name"})])

        # Transform
        X = preprocessor.transform(input_df)

        # Get model
        model_name = data.model_name.lower()
        model = models.get(model_name, models["randomforest"])

        # Predict
        y_pred = model.predict(X)

        return {
            "prediction": float(y_pred[0]),
            "model": model_name
        }

    except Exception as e:
        return {"error": str(e)}