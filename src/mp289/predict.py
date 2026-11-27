"""Predict the holdout file with the champion model (PROBLEM.md section 8)."""

from pathlib import Path

import pandas as pd

from mp289.data import clean_common


def predict_holdout(champion, hold: pd.DataFrame, features) -> pd.DataFrame:
    """One row per holdout record: record_id and the predicted fare."""
    frame = clean_common(hold)
    preds = champion.predict(frame[list(features)])
    return pd.DataFrame({
        "record_id": hold["record_id"].astype(str),
        "prediction": pd.Series(preds).round(2),
    })


def write_predictions(predictions: pd.DataFrame, path) -> Path:
    """Write record_id,prediction and nothing else."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path
