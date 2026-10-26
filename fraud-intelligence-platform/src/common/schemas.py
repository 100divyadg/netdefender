"""Column contract shared by training and (later) streaming inference."""

TARGET = "isFraud"

# Every column the training script needs from the raw CSV.
RAW_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
REQUIRED_COLUMNS = RAW_FEATURE_COLUMNS + [TARGET]

# Present in the CSV but deliberately never used as model inputs:
# isFlaggedFraud is an existing rule's output, the names are identifiers.
EXCLUDED_COLUMNS = ["isFlaggedFraud", "nameOrig", "nameDest"]

# Some PaySim copies spell the origin balance column differently.
COLUMN_ALIASES = {"oldbalanceOrig": "oldbalanceOrg", "newbalanceOrg": "newbalanceOrig"}

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Model inputs, in the exact order the preprocessor sees them.
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "hour_of_day",
    "origin_balance_error",
    "dest_balance_change",
]
CATEGORICAL_FEATURES = ["type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
