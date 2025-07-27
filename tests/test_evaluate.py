import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import evaluate
import numpy as np

def test_evaluate_model_outputs_metrics():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    class DummyModel:
        def predict(self, X): return np.random.rand(len(X))

    metrics = evaluate.evaluate_model(DummyModel(), X, y)
    assert all(k in metrics for k in ("rmse", "mae", "r2"))
