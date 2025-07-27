import pandas as pd
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "women_in_stem.csv"
REPORT_DIR = BASE_DIR / "monitoring" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = (
    df.columns.str.lower()
        .str.replace(r"\s*\([^)]*\)", "", regex=True)
        .str.replace(" ", "_")
        .str.strip("_")
)

# Split data into reference and production
df_ref = df[df["year"] < 2012].copy()
df_prod = df[df["year"] >= 2012].copy()

# --- Test Suite: data stability ---
stability_suite = TestSuite(tests=[
    DataStabilityTestPreset(),
])
stability_suite.run(reference_data=df_ref, current_data=df_prod)
stability_suite.save_html(str(REPORT_DIR / "data_stability.html"))

# --- Report: data drift ---
drift_report = Report(metrics=[
    DataDriftPreset(),
])
drift_report.run(reference_data=df_ref, current_data=df_prod)
drift_report.save_html(str(REPORT_DIR / "data_drift.html"))

print("Reports saved:")
print(f"Data Stability → {REPORT_DIR / 'data_stability.html'}")
print(f"Data Drift     → {REPORT_DIR / 'data_drift.html'}")