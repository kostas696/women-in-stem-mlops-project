import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import os

# Project structure
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = BASE_DIR / "data" / "raw" / "women_in_stem.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# ---------- PART 1: LOAD AND CLEAN ----------

def load_and_clean_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = (
        df.columns.str.lower()
        .str.replace(r"\s*\([^)]*\)", "", regex=True)
        .str.replace(" ", "_")
        .str.strip("_")
    )
    return df

# ---------- PART 2: ENCODE AND TRANSFORM ----------

def encode_and_transform(df: pd.DataFrame):
    features = ["year", "female_enrollment", "gender_gap_index", "country", "stem_fields"]
    target = "female_graduation_rate"

    X = df[features]
    y = df[target]

    cat_features = ["country", "stem_fields"]
    num_features = ["year", "female_enrollment", "gender_gap_index"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

# ---------- PART 3: SAVE CLEANED DATA AND PREPROCESSOR ----------

def save_artifacts(X, y, preprocessor):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")

# ---------- PART 4: ENTRY POINT ----------

def main():
    df = load_and_clean_data(RAW_DATA)
    X, y, preprocessor = encode_and_transform(df)
    save_artifacts(X, y, preprocessor)

if __name__ == "__main__":
    main()