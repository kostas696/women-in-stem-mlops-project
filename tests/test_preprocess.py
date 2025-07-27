import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import preprocess
import numpy as np

def test_preprocess_pipeline_outputs_valid_shapes():
    df = preprocess.load_and_clean_data("data/raw/women_in_stem.csv")
    X, y, preprocessor = preprocess.encode_and_transform(df)
    assert X.shape[0] == len(y)
    assert X.shape[1] > 0
