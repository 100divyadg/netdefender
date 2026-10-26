"""Train and evaluate the baseline XGBoost fraud classifier.

Usage: python -m src.training.train [--data path.csv] [--artifacts-dir dir] [--nrows N]

Writes:
  artifacts/fraud_model.joblib   preprocessing + classifier as one sklearn Pipeline
  artifacts/model_metadata.json  version, feature schema, threshold, metrics
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.common.config import (
    ARTIFACT_DIR,
    DATA_PATH,
    METADATA_FILENAME,
    MODEL_FILENAME,
    RANDOM_STATE,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from src.common.schemas import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
)
from src.training.features import (
    build_preprocessor,
    normalise_columns,
    to_model_input,
    validate_training_frame,
)

NUMERIC_DTYPES = {c: "float64" for c in REQUIRED_COLUMNS if c not in ("step", "type", TARGET)}


def load_dataset(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Pass --data or set FRAUD_DATA_PATH to your downloaded CSV."
        )
    df = normalise_columns(pd.read_csv(path, nrows=nrows))
    validate_training_frame(df)
    return df[REQUIRED_COLUMNS].astype(NUMERIC_DTYPES)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on `step` boundaries so no time step appears in two splits."""
    ordered = df.sort_values("step", kind="stable")
    n = len(ordered)
    train_cut = ordered["step"].iloc[int(n * TRAIN_FRACTION) - 1]
    val_cut = ordered["step"].iloc[int(n * (TRAIN_FRACTION + VALIDATION_FRACTION)) - 1]

    train = ordered[ordered["step"] <= train_cut]
    val = ordered[(ordered["step"] > train_cut) & (ordered["step"] <= val_cut)]
    test = ordered[ordered["step"] > val_cut]

    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty or part[TARGET].nunique() < 2:
            raise ValueError(
                f"The {name} split has {len(part)} rows and {part[TARGET].sum()} fraud cases; "
                "it needs both classes. Use more rows or check the step distribution."
            )
    return train, val, test


def build_model(scale_pos_weight: float) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])


def select_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Pick the probability threshold that maximises F1 on the validation split."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The final precision/recall pair has no threshold attached.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def metrics_at(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_split(name: str, part: pd.DataFrame) -> dict:
    info = {
        "rows": len(part),
        "fraud": int(part[TARGET].sum()),
        "step_min": int(part["step"].min()),
        "step_max": int(part["step"].max()),
    }
    print(f"{name:<11} rows={info['rows']:>9,}  fraud={info['fraud']:>6,}  steps {info['step_min']}-{info['step_max']}")
    return info


def print_metrics(title: str, m: dict) -> None:
    cm = m["confusion_matrix"]
    print(f"\n{title} (threshold {m['threshold']:.4f})")
    print(f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  fpr={m['false_positive_rate']:.6f}")
    print(f"  confusion matrix: TN={cm['tn']:,}  FP={cm['fp']:,}  FN={cm['fn']:,}  TP={cm['tp']:,}")


def train(data_path: Path, artifact_dir: Path, nrows: int | None = None) -> dict:
    print(f"Loading {data_path}")
    df = load_dataset(data_path, nrows)
    train_df, val_df, test_df = chronological_split(df)

    splits = {
        "train": describe_split("train", train_df),
        "validation": describe_split("validation", val_df),
        "test": describe_split("test", test_df),
    }

    X_train, y_train = to_model_input(train_df), train_df[TARGET]
    X_val, y_val = to_model_input(val_df), val_df[TARGET]
    X_test, y_test = to_model_input(test_df), test_df[TARGET]

    negatives, positives = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = negatives / positives
    model = build_model(scale_pos_weight)

    print(f"\nTraining XGBoost on {len(X_train):,} rows (scale_pos_weight={scale_pos_weight:.1f})")
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])

    started = time.perf_counter()
    test_proba = model.predict_proba(X_test)[:, 1]
    batch_seconds = time.perf_counter() - started

    # Streaming scores small micro-batches, so also time a single event end to end.
    single_row = X_test.iloc[[0]]
    single_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict_proba(single_row)
        single_times.append(time.perf_counter() - t0)

    test_metrics = {
        "pr_auc": round(float(average_precision_score(y_test, test_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_proba)), 4),
        "at_default_0_5": metrics_at(y_test, test_proba, 0.5),
        "at_selected_threshold": metrics_at(y_test, test_proba, threshold),
    }
    latency = {
        "train_seconds": round(train_seconds, 2),
        "batch_rows": len(X_test),
        "batch_ms_per_1000_rows": round(batch_seconds / len(X_test) * 1e6, 3),
        "single_event_ms_median": round(float(np.median(single_times)) * 1e3, 3),
    }

    print(f"\nTest PR-AUC: {test_metrics['pr_auc']:.4f}   ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print_metrics("Test @ default threshold", test_metrics["at_default_0_5"])
    print_metrics("Test @ threshold chosen on validation", test_metrics["at_selected_threshold"])
    print(
        f"\nInference: {latency['batch_ms_per_1000_rows']:.3f} ms per 1,000 rows (batch), "
        f"{latency['single_event_ms_median']:.3f} ms median per single event"
    )

    trained_at = datetime.now(timezone.utc)
    metadata = {
        "model_version": trained_at.strftime("v%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(),
        "dataset": {
            "file": data_path.name,
            "sha256": file_sha256(data_path),
            "rows_used": len(df),
            "nrows_limit": nrows,
        },
        "target": TARGET,
        "feature_schema": {
            "model_features": MODEL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "excluded_columns": EXCLUDED_COLUMNS,
            "encoded_features": model.named_steps["preprocess"].get_feature_names_out().tolist(),
        },
        "decision_threshold": round(threshold, 6),
        "threshold_selection": "max F1 on chronological validation split",
        "splits": splits,
        "hyperparameters": {
            k: v
            for k, v in model.named_steps["classifier"].get_params().items()
            if k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "scale_pos_weight")
        },
        "test_metrics": test_metrics,
        "latency": latency,
        "library_versions": {
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / MODEL_FILENAME)
    (artifact_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved {artifact_dir / MODEL_FILENAME} and {METADATA_FILENAME} ({metadata['model_version']})")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the PaySim CSV")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACT_DIR, help="Where to write the model")
    parser.add_argument("--nrows", type=int, default=None, help="Read only the first N rows (quick experiments)")
    args = parser.parse_args()
    train(args.data, args.artifacts_dir, args.nrows)


if __name__ == "__main__":
    main()
