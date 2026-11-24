import pandas as pd

from mp289.data import clean_booking, clean_train

P = {
    "target": "fare_rs",
    "typo_threshold": 1000,
    "unit_fix": {"column": "distance_km", "zone": "Outskirts", "factor": 1000},
}


def test_clean_booking_levels():
    s = pd.Series([" stand", "Stand", "STAND", "sta.", "str.", "Street ", "APP", "app."])
    assert set(clean_booking(s)) == {"stand", "street", "app"}


def test_clean_train_fixes_unit_typo_and_leak():
    df = pd.DataFrame({
        "distance_km": [5000.0, 10.0],
        "zone": ["Outskirts", "Market"],
        "booking": ["sta.", "app"],
        "fare_rs": [2500, 250],
        "fare_paid_receipt_rs": [240, 250],
    })
    out = clean_train(df, P)
    assert out["distance_km"].tolist() == [5.0, 10.0]
    assert out["fare_rs"].tolist() == [250.0, 250.0]
    assert "fare_paid_receipt_rs" not in out.columns
    assert out["booking"].tolist() == ["stand", "app"]


def test_clean_train_drops_duplicates():
    row = {"distance_km": 5.0, "zone": "Market", "booking": "app",
           "fare_rs": 200, "fare_paid_receipt_rs": 200}
    out = clean_train(pd.DataFrame([row, row]), P)
    assert len(out) == 1