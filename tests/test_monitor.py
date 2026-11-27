import numpy as np
import pandas as pd
import yaml

from mp289.monitor import psi, psi_table, should_promote, time_split, update_findings
from mp289.predict import predict_holdout


def test_psi_is_zero_for_a_column_against_itself():
    s = pd.Series(np.linspace(0, 100, 500))
    assert psi(s, s) < 1e-6


def test_psi_grows_with_the_size_of_the_shift():
    base = pd.Series(np.linspace(0, 100, 500))
    assert 0 < psi(base, base + 2) < psi(base, base + 15)


def test_psi_counts_values_outside_the_training_range():
    base = pd.Series(np.linspace(1, 10, 100))
    wide = pd.Series([0.0] * 50 + [50.0] * 50)
    value = psi(base, wide)
    assert np.isfinite(value) and value > 0


def test_psi_table_covers_every_numeric_column():
    rng = np.random.default_rng(0)
    columns = ["distance_km", "wait_min", "passengers", "hour", "luggage_bags", "fuel_price_rs"]
    train = pd.DataFrame({c: rng.normal(10, 1, 200) for c in columns})
    later = pd.DataFrame({c: rng.normal(20, 2, 200) for c in columns})
    table = psi_table(train, later)
    assert set(table) == set(columns)
    assert all(value > 0 for value in table.values())


def test_time_split_sorts_by_date_and_keeps_rows_apart():
    df = pd.DataFrame({
        "date": ["2026-08-03", "2026-07-01", "2026-08-01", "2026-07-02"],
        "record_id": ["d", "a", "c", "b"],
    })
    fit, check = time_split(df, 0.7)
    assert list(fit["record_id"]) == ["a", "b", "c"]
    assert list(check["record_id"]) == ["d"]


def test_should_promote_needs_twelve_percent():
    assert should_promote(40.0, 35.2) is True
    assert should_promote(40.0, 35.3) is False
    assert should_promote(40.0, 50.0) is False


def test_update_findings_keeps_the_header_and_fills_monitoring(tmp_path):
    path = tmp_path / "findings.yaml"
    path.write_text(
        '# keep this comment\ngroup_id: "G188"\ntracking:\n  champion_version: 1\n\n'
        "monitoring:\n  psi:\n    distance_km: null\n  decision: null\n",
        encoding="utf-8",
    )
    monitoring = {
        "psi": {"distance_km": 0.12},
        "most_drifted_feature": "distance_km",
        "decision": "keep",
    }
    text = update_findings(path, monitoring, champion_version=2)
    loaded = yaml.safe_load(text)
    assert text.startswith("# keep this comment")
    assert loaded["group_id"] == "G188"
    assert loaded["tracking"]["champion_version"] == 2
    assert loaded["monitoring"]["psi"]["distance_km"] == 0.12
    assert loaded["monitoring"]["decision"] == "keep"


class _Double:
    """Predict twice the distance, so the test needs no trained model."""

    def predict(self, X):
        return X["distance_km"].to_numpy() * 2


def test_predict_holdout_has_two_columns_and_one_row_per_record():
    hold = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "distance_km": [1.0, 2.5],
        "booking": ["Street ", "STA."],
        "zone": ["Market", "Outskirts"],
    })
    out = predict_holdout(_Double(), hold, ["distance_km"])
    assert list(out.columns) == ["record_id", "prediction"]
    assert out["prediction"].tolist() == [2.0, 5.0]
