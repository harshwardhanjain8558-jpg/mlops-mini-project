import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge

import app.main as main
from mp289.data import clean_booking
from mp289.train import build_pipeline

GOOD = {
    "distance_km": 5.2, "wait_min": 3.0, "passengers": 2, "hour": 9,
    "luggage_bags": 1, "fuel_price_rs": 104.5, "booking": "app", "zone": "Market",
}


@pytest.fixture
def client(monkeypatch):
    """Tiny model trained on made-up rows, so tests never need the CSVs."""
    rows = pd.DataFrame([
        {**GOOD, "distance_km": d, "booking": b, "zone": z, "fare_rs": 30 + 12 * d}
        for d in [1, 3, 5, 8, 12]
        for b in ["app", "street", "stand"]
        for z in ["Market", "Station Area", "Outskirts"]
    ])
    pipe = build_pipeline(Ridge()).fit(rows[main.FEATURES], rows["fare_rs"])
    monkeypatch.setattr(main, "model", pipe)
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["fare_rs"] > 0


def test_fuel_price_above_150_rejected(client):
    assert client.post("/predict", json={**GOOD, "fuel_price_rs": 150.01}).status_code == 422
    assert client.post("/predict", json={**GOOD, "fuel_price_rs": 150}).status_code == 200


def test_negative_distance_rejected(client):
    assert client.post("/predict", json={**GOOD, "distance_km": -0.1}).status_code == 422
    assert client.post("/predict", json={**GOOD, "distance_km": 0}).status_code == 200


def test_missing_or_extra_field_rejected(client):
    missing = {k: v for k, v in GOOD.items() if k != "zone"}
    assert client.post("/predict", json=missing).status_code == 422
    leak = {**GOOD, "fare_paid_receipt_rs": 99}
    assert client.post("/predict", json=leak).status_code == 422


def test_messy_booking_spelling_is_cleaned(client):
    a = client.post("/predict", json={**GOOD, "booking": "Street "}).json()
    b = client.post("/predict", json={**GOOD, "booking": "street"}).json()
    assert a == b


def test_clean_booking():
    s = pd.Series(["Street ", "STA.", "app", "str"])
    assert clean_booking(s).tolist() == ["street", "stand", "app", "street"]
