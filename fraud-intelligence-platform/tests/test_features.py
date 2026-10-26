import pandas as pd
import pytest

from src.common.schemas import EXCLUDED_COLUMNS, MODEL_FEATURES, TARGET
from src.training.features import (
    add_engineered_features,
    build_preprocessor,
    normalise_columns,
    to_model_input,
    validate_training_frame,
)


def test_engineered_features_match_formulas():
    row = pd.DataFrame(
        [{"step": 25, "amount": 100.0, "oldbalanceOrg": 300.0, "newbalanceOrig": 150.0,
          "oldbalanceDest": 10.0, "newbalanceDest": 110.0}]
    )
    out = add_engineered_features(row).iloc[0]
    assert out["hour_of_day"] == 1
    assert out["origin_balance_error"] == 50.0
    assert out["dest_balance_change"] == 100.0


def test_engineering_does_not_mutate_input(paysim_df):
    before = paysim_df.copy()
    add_engineered_features(paysim_df)
    pd.testing.assert_frame_equal(paysim_df, before)


def test_model_input_has_fixed_order_and_no_leaky_columns(paysim_df):
    X = to_model_input(paysim_df)
    assert X.columns.tolist() == MODEL_FEATURES
    for leaky in [TARGET, "step", *EXCLUDED_COLUMNS]:
        assert leaky not in X.columns


def test_preprocessor_layout_is_stable_and_handles_unseen_type(paysim_df):
    pre = build_preprocessor().fit(to_model_input(paysim_df))
    names = pre.get_feature_names_out().tolist()
    assert names[-5:] == ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

    unseen = to_model_input(paysim_df.head(1).assign(type="CRYPTO"))
    encoded = pre.transform(unseen)
    assert encoded[0, -5:].sum() == 0


def test_column_aliases_are_normalised(paysim_df):
    renamed = paysim_df.rename(columns={"oldbalanceOrg": "oldbalanceOrig"})
    assert "oldbalanceOrg" in normalise_columns(renamed).columns


def test_validation_rejects_missing_columns(paysim_df):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_training_frame(paysim_df.drop(columns=["amount"]))


def test_validation_rejects_non_binary_target(paysim_df):
    with pytest.raises(ValueError, match="must be 0/1"):
        validate_training_frame(paysim_df.assign(isFraud=2))


def test_validation_rejects_single_class(paysim_df):
    with pytest.raises(ValueError, match="single class"):
        validate_training_frame(paysim_df.assign(isFraud=0))
