import pytest
from fastapi.testclient import TestClient

import app.main as main

GOOD = {
    "distance_km": 10.5, "wait_min": 5, "passengers": 2, "hour": 14,
    "luggage_bags": 1, "fuel_price_rs": 100.0,
    "booking": "app", "zone": "Market",
}


class FakeModel:
    def predict(self, X):
        return [123.4] * len(X)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "_model", FakeModel())
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["prediction"] == pytest.approx(123.4)


def test_rejects_fuel_price_above_150(client):
    assert client.post("/predict", json={**GOOD, "fuel_price_rs": 150.5}).status_code == 422


def test_rejects_negative_distance(client):
    assert client.post("/predict", json={**GOOD, "distance_km": -1}).status_code == 422


def test_rejects_missing_field(client):
    bad = {k: v for k, v in GOOD.items() if k != "zone"}
    assert client.post("/predict", json=bad).status_code == 422