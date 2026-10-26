"""Central configuration. Every phase reads paths and thresholds from here."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = Path(os.getenv("FRAUD_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "paysim.csv"))
ARTIFACT_DIR = Path(os.getenv("FRAUD_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))

MODEL_FILENAME = "fraud_model.joblib"
METADATA_FILENAME = "model_metadata.json"

# Chronological split by `step`: the first 70% of rows train, the next 15%
# choose the decision threshold, the final 15% are the untouched test set.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

RANDOM_STATE = 42
