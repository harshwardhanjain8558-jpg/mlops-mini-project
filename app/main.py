import json
import logging
import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
FEATURES = ["distance_km", "wait_min", "passengers", "hour",
            "luggage_bags", "fuel_price_rs", "booking", "zone"]


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": self.formatTime(record), "level": record.levelname,
             "msg": record.getMessage()}
        d.update(getattr(record, "data", {}))
        return json.dumps(d)


logger = logging.getLogger("mp289")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class PredictRequest(BaseModel):
    distance_km: float = Field(..., ge=0)
    wait_min: float = Field(..., ge=0)
    passengers: int = Field(..., ge=1)
    hour: int = Field(..., ge=0, le=23)
    luggage_bags: int = Field(..., ge=0)
    fuel_price_rs: float = Field(..., le=150)
    booking: Literal["street", "app", "stand"]
    zone: Literal["Station Area", "Market", "Outskirts"]


app = FastAPI(title="MP289 fare API")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.model_dump()])[FEATURES]
    pred = float(get_model().predict(df)[0])
    logger.info("prediction", extra={"data": {**req.model_dump(), "prediction": pred}})
    return {"prediction": pred}