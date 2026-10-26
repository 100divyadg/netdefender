import json

import joblib

from src.common.config import METADATA_FILENAME, MODEL_FILENAME
from src.common.schemas import MODEL_FEATURES
from src.training.features import to_model_input
from src.training.train import chronological_split, load_dataset, train


def test_chronological_split_does_not_share_steps(paysim_csv):
    tr, va, te = chronological_split(load_dataset(paysim_csv))
    assert tr["step"].max() < va["step"].min()
    assert va["step"].max() < te["step"].min()
    assert len(tr) + len(va) + len(te) == len(load_dataset(paysim_csv))


def test_training_writes_usable_artifacts(paysim_csv, paysim_df, tmp_path):
    out_dir = tmp_path / "artifacts"
    metadata = train(paysim_csv, out_dir)

    saved = json.loads((out_dir / METADATA_FILENAME).read_text())
    assert saved["model_version"] == metadata["model_version"]
    assert saved["feature_schema"]["model_features"] == MODEL_FEATURES
    assert 0 < saved["decision_threshold"] < 1

    model = joblib.load(out_dir / MODEL_FILENAME)
    proba = model.predict_proba(to_model_input(paysim_df))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    # The synthetic fraud pattern is trivially learnable; this guards against a broken pipeline.
    assert saved["test_metrics"]["pr_auc"] > 0.9
