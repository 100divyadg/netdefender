"""Read-only data quality report for the PaySim CSV.

Usage: python -m src.training.inspect_data [--data path/to/file.csv]
"""

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import DATA_PATH
from src.common.schemas import REQUIRED_COLUMNS, TARGET
from src.training.features import normalise_columns


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def inspect(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Pass --data or set FRAUD_DATA_PATH to your downloaded CSV."
        )

    df = pd.read_csv(path)
    original_columns = df.columns.tolist()
    df = normalise_columns(df)

    section("File")
    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size / 1e6:.1f} MB")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    section("Columns")
    print(original_columns)
    if original_columns != df.columns.tolist():
        print("Renamed to project names:", df.columns.tolist())
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    print("Required columns missing:", missing or "none")

    section("Data types")
    print(df.dtypes.to_string())

    section("Missing values")
    print(df.isnull().sum().sort_values(ascending=False).to_string())

    section("Duplicates")
    print("Duplicate rows:", int(df.duplicated().sum()))

    if TARGET in df.columns:
        section("Fraud distribution")
        counts = df[TARGET].value_counts(dropna=False)
        pct = df[TARGET].value_counts(normalize=True, dropna=False).mul(100).round(4)
        print(pd.DataFrame({"count": counts, "percent": pct}).to_string())

        if "type" in df.columns:
            section("Fraud by transaction type")
            by_type = df.groupby("type")[TARGET].agg(transactions="count", fraud="sum")
            by_type["fraud_rate_pct"] = (by_type["fraud"] / by_type["transactions"] * 100).round(4)
            print(by_type.to_string())

    if "step" in df.columns:
        section("Time range")
        print(f"step: min={df['step'].min()}, max={df['step'].max()}, unique={df['step'].nunique()}")

    section("Numeric summary")
    print(df.describe().T.to_string())

    section("Sample records")
    print(df.head().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the PaySim CSV")
    inspect(parser.parse_args().data)


if __name__ == "__main__":
    main()
