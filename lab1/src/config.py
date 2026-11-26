"""
Configuration and constants for SEC-635 Lab 1 (AI-Driven Detection).
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parents[1]  # the lab1/ directory
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORT_DIR = BASE_DIR / "report"

# Ensure output directories exist
for d in [DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset filenames
WEDNESDAY_CSV = DATA_DIR / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
THURSDAY_CSV = DATA_DIR / "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"

WEDNESDAY_PARQUET = DATA_DIR / "cleaned_wednesday_14-02-2018.parquet"
THURSDAY_PARQUET = DATA_DIR / "cleaned_thursday_01-03-2018.parquet"

# Metadata & Operational Constants
RANDOM_STATE = 42
CAPTURE_DURATION_HOURS = 12.0   # Each day represents a 12-hour capture
DAY_HOURS = 24.0                # 24-hour SOC day
ANALYST_TRIAGE_MINUTES = 12.0   # Average time for a Tier-1 analyst to triage an alert

# Features to drop due to data leakage / static identifier memorization
# "Dst Port" leaks because target ports (21, 22) uniquely correlate with attack labels in testbed.
LEAKAGE_COLUMNS = [
    "Dst Port",
    "Timestamp",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port"
]

# Labels mapping
LABEL_COL = "Label"
BENIGN_LABEL = "Benign"

# Temporal split fraction
TEMPORAL_TRAIN_FRACTION = 0.70  # First 70% of chronological time for training, last 30% for testing
