"""Feature engineering shared by offline training and streaming inference.

Everything here must be computable from a single transaction event at the
moment it arrives, so the streaming path can call the same functions.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.common.schemas import (
    CATEGORICAL_FEATURES,
    COLUMN_ALIASES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
    TRANSACTION_TYPES,
)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column-name variants to the names used in this project."""
    renames = {old: new for old, new in COLUMN_ALIASES.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=renames)


def validate_training_frame(df: pd.DataFrame) -> None:
    """Fail loudly if the dataset does not match the expected contract."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {df.columns.tolist()}")

    labels = set(df[TARGET].dropna().unique())
    if not labels <= {0, 1}:
        raise ValueError(f"{TARGET} must be 0/1, found values {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError(f"{TARGET} contains a single class; cannot train a classifier")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Required columns contain nulls: {null_counts[null_counts > 0].to_dict()}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the derived model inputs added.

    hour_of_day replaces raw `step`: step only grows over the simulation, so a
    chronological test set would contain values the model never saw. The hour
    keeps the daily cycle without that extrapolation problem.
    """
    out = df.copy()
    out["hour_of_day"] = out["step"] % 24
    out["origin_balance_error"] = out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
    out["dest_balance_change"] = out["newbalanceDest"] - out["oldbalanceDest"]
    return out


def to_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and select them in the fixed model order."""
    return add_engineered_features(df)[MODEL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    # Fixed categories keep the encoded layout identical across retrains;
    # an unseen type at inference time encodes to all zeros instead of crashing.
    encoder = OneHotEncoder(categories=[TRANSACTION_TYPES], handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("type", encoder, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
