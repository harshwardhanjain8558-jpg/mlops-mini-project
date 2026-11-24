import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mp289.data import ROOT, clean_common, load_params

PARAMS = load_params()
FEATURES = PARAMS["features"]
MODEL_PATH = ROOT / os.getenv("MODEL_PATH", "models/model.joblib")

model = None  # set at startup (or by tests)


class Ride(BaseModel):
    """One ride request. Only fields known *before* the ride starts."""

    model_config = ConfigDict(extra="forbid")  # exactly these fields, nothing else

    distance_km: float = Field(..., ge=0)       # client rule 2: below 0 is a typo
    wait_min: float
    passengers: int
    hour: int
    luggage_bags: int
    fuel_price_rs: float = Field(..., le=150)   # client rule 1: above 150 impossible
    booking: str
    zone: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if model is None and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="MP289 auto-rickshaw fare", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(ride: Ride):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    df = clean_common(pd.DataFrame([ride.model_dump()]))
    fare = float(model.predict(df[FEATURES])[0])
    return {"fare_rs": round(fare, 2)}
