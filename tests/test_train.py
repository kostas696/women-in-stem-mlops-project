import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import train

def test_model_factory():
    rf = train.get_model("randomforest")
    xgb = train.get_model("xgboost")
    cb = train.get_model("catboost")

    assert rf.__class__.__name__ == "RandomForestRegressor"
    assert xgb.__class__.__name__ == "XGBRegressor"
    assert cb.__class__.__name__ == "CatBoostRegressor"
