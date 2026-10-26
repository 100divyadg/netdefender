# Real-Time Fraud Detection & Transaction Intelligence Platform

A streaming fraud-detection prototype built on the synthetic **PaySim** mobile-money
dataset: Kafka → Spark Structured Streaming → XGBoost → PostgreSQL / Redis → FastAPI.

This is an educational prototype trained on synthetic data. It is not production-ready.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Inspect the dataset | **Done**: `src/training/inspect_data.py` |
| 2 | Baseline XGBoost model | **Done**: `src/training/train.py` (still needs a run on the real PaySim CSV) |
| 3 | Kafka producer / topics | Not started |
| 4 | Spark streaming inference | Not started |
| 5 | PostgreSQL, FastAPI, Redis | Not started |
| 6 | Docker, CI, load tests | Not started |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Put the PaySim CSV at `data/raw/paysim.csv`. The Kaggle download is named
`PS_20174392719_1491204439457_log.csv`; rename it, pass `--data`, or set
`FRAUD_DATA_PATH`. The CSV is git-ignored and must not be committed.

## Phase 1: inspect the data

```bash
python -m src.training.inspect_data            # or: --data path/to/file.csv
```

This is read-only. It prints file size, shape, columns (and any missing required ones),
dtypes, nulls, duplicates, fraud rate overall and per transaction type, the `step` range,
a numeric summary and sample rows.

## Phase 2: train the baseline

```bash
python -m src.training.train                   # --nrows 500000 for a quick run
```

Outputs, both git-ignored:

- `artifacts/fraud_model.joblib`: one sklearn `Pipeline` (preprocessing + XGBoost), so
  inference cannot drift from the training-time encoding.
- `artifacts/model_metadata.json`: model version, dataset SHA-256, feature schema, chosen
  threshold, split sizes, hyperparameters, test metrics, latency and library versions.

### Method

**Target:** `isFraud`. **Never used as inputs:** `isFraud`, `isFlaggedFraud` (an existing
rule's output), `nameOrig` and `nameDest` (identifiers).

**Features** (`src/common/schemas.py`, built in `src/training/features.py`):

| Feature | Definition |
|---|---|
| `type` | One-hot encoded with a fixed category list; an unseen type encodes to all zeros |
| `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | Raw |
| `hour_of_day` | `step % 24` |
| `origin_balance_error` | `oldbalanceOrg - amount - newbalanceOrig` |
| `dest_balance_change` | `newbalanceDest - oldbalanceDest` |

Raw `step` is replaced by `hour_of_day`. `step` only increases over the simulation, so in a
chronological split every test value lies beyond anything seen in training and the trees
would just treat it as "the largest step seen". The hour keeps the daily pattern without
that problem.

**Split:** chronological by `step`, 70% train / 15% validation / 15% test, with cuts placed
on step boundaries so no step appears in two splits. Known limitation: the same account can
still appear in more than one split.

**Imbalance:** `scale_pos_weight = negatives / positives` on the training split.

**Threshold:** chosen on the validation split by maximising F1, then applied unchanged to the
test split. Test metrics are reported at both 0.5 and the chosen threshold.

**Metrics recorded:** PR-AUC, ROC-AUC, precision, recall, F1, false-positive rate, confusion
matrix, batch and single-event inference latency.

### Results

_Not yet recorded._ Paste the output of `python -m src.training.train` on the full PaySim
file here. Don't copy numbers from elsewhere.

## Tests

```bash
pytest
```

The tests build a small synthetic PaySim-shaped frame, so they run without the real dataset.
They cover the feature formulas, column order and leak exclusion, encoder stability, the
dataset validation errors, a chronological split with no shared steps, and an end-to-end
training run that loads the saved artifact and checks that probabilities fall in [0, 1].

## Layout

```
src/common/config.py        paths (env-overridable), split fractions
src/common/schemas.py       column contract shared by training and streaming
src/training/features.py    validation, feature engineering, preprocessor
src/training/inspect_data.py
src/training/train.py
tests/
```

Only load `.joblib` model files that you created yourself. Unpickling can run arbitrary code.
