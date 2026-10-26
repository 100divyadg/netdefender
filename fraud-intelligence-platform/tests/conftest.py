import numpy as np
import pandas as pd
import pytest

from src.common.schemas import TRANSACTION_TYPES


def make_paysim_like(n: int = 6000, steps: int = 60, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with PaySim's columns and a learnable fraud pattern."""
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, steps + 1, n))
    tx_type = rng.choice(TRANSACTION_TYPES, n)
    old_org = rng.uniform(0, 200_000, n).round(2)
    amount = (old_org * rng.uniform(0.01, 0.5, n)).round(2)
    old_dest = rng.uniform(0, 100_000, n).round(2)

    # Fraud: TRANSFER / CASH_OUT that drains the whole origin balance.
    is_fraud = (np.isin(tx_type, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.08)).astype(int)
    amount = np.where(is_fraud == 1, old_org, amount)
    new_org = (old_org - amount).round(2)

    return pd.DataFrame(
        {
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "nameDest": [f"M{i}" for i in range(n)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": (old_dest + amount).round(2),
            "isFraud": is_fraud,
            "isFlaggedFraud": 0,
        }
    )


@pytest.fixture
def paysim_df() -> pd.DataFrame:
    return make_paysim_like()


@pytest.fixture
def paysim_csv(tmp_path, paysim_df):
    path = tmp_path / "paysim.csv"
    paysim_df.to_csv(path, index=False)
    return path
