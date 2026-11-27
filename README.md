# MP289 — Auto-Rickshaw Fare (Dehradun)

**Group:** G188 · **Problem:** MP289 · **Faculty:** Richa Sharma · **Batch:** 87

| Enrolment no. | Member |
|---|---|
| S24CSEU2299 | RishabhVerma |
| S24CSEU2339 | VINAMRAAGARWAL |
| S24CSEU2328 | Harshwardhanjain |
| S24CSEU2322 | UtkarshKumar |

A regression service for the Dehradun Auto Drivers' Cooperative: predict `fare_rs`
for a ride from the features known *before* the ride starts, so riders see the
price before they book.

## The data

| File | Rows | What it is |
|---|---|---|
| `data/raw/train.csv` | 1000 | Jan–Jun 2026, messy — tracked with DVC (`train.csv.dvc`) |
| `data/raw/later.csv` | 300 | Jul–Aug 2026, clean — tracked with DVC (`later.csv.dvc`) |
| `data/raw/holdout_features.csv` | 250 | Sep 2026, no target — the file we predict |

`train.csv` and `later.csv` are not committed (see `.gitignore`); their DVC files
pin the hashes.

## What the data checks found

All seven checks from the handbook, coded in `src/mp289/checks.py`:

| Problem | Present | Details |
|---|---|---|
| `leaking_column` | yes | `fare_paid_receipt_rs` — only filled in after the ride, absent from the holdout, and within ₹5 of the target on 929/1000 rows (r = 0.999 once the typo rows are excluded) |
| `unit_mixup` | yes | `distance_km` for zone `Outskirts` is in **metres** — mean 13 144 vs 13.1 for `Market`, a factor of **1000** |
| `sentinel_value` | no | no `-1` / `-999` / `999` / `9999` anywhere, no negative readings |
| `duplicate_rows` | no | `df.duplicated().sum() == 0` |
| `stuck_sensor` | no | longest constant run in date order is 6 rows over 1 day; `fuel_price_rs` takes 802 distinct values |
| `target_typos` | yes | **11** targets typed with an extra zero (1660 … 3760, with a gap from 456); dividing by 10 matches the receipt |
| `spelling_variants` | yes | `booking` has 18 spellings (`stand`, `Stand`, `stand `, `sta.`, `STAND`, `str.`, …) of **3** real levels; `later.csv` has the clean ones |

Cleaning (`src/mp289/data.py`) drops the receipt column, normalises `booking`,
divides `Outskirts` distance by 1000, divides the typo targets by 10 and removes
duplicates.

## Pipeline

```
src/mp289/data.py      load params, clean train / later / holdout
src/mp289/train.py     build the sklearn pipeline, train, log to MLflow, register
src/mp289/checks.py    the seven data checks (evidence for findings.yaml)
src/mp289/monitor.py   PSI, MAE per zone, retrain v2, promotion rule, findings
src/mp289/predict.py   predict the holdout file with the champion
src/mp289/logs.py      JSON logging used by the API
app/main.py            FastAPI service: POST /predict, GET /health
```

## Results

- **v1** (Ridge, cleaned `train.csv`): `mae_cv` 10.67, `mae_later` **41.85** —
  run `6dec3ed13de849be99248d2a38d33171`, registered as version 1.
- **Drift** (PSI, 10 bins, cleaned train vs `later.csv`): `fuel_price_rs`
  **3.65** is by far the most drifted column (next highest: `hour` 0.058).
- **v1 per zone** on `later.csv`: Market **109.19** worst, Station Area 24.65,
  Outskirts 22.88.
- **v2** (same Ridge, trained on cleaned `train.csv` + first 70% of
  `later.csv` by date): **26.90** vs v1's **41.18** on the last 30% — a
  **34.67%** drop, over the 12% bar, so **v2 was promoted** to `@champion`
  (version 2). `models/model.joblib` is the champion the API loads.
- `predictions/holdout_predictions.csv` — 250 rows, `record_id,prediction`.

All numbers above come from the code in this repository running on these files;
`reports/findings.yaml` holds them in the form the marking tool reads.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

dvc pull                        # or copy train.csv / later.csv into data/raw/
python -m mp289.train           # v1 + experiments (needs PYTHONPATH=src)
python -m mp289.monitor         # drift, retrain, promotion, predictions, findings
python -m mp289.checks          # the seven data checks
pytest -q                       # tests build their own small data, no CSVs needed
uvicorn app.main:app --port 8000
```

Set `PYTHONPATH=src` (the Dockerfile and `pytest.ini` do this for you). The
service needs `models/model.joblib`; `mlflow.db` holds the runs and the registry.

## Service

`POST /predict` takes exactly `distance_km`, `wait_min`, `passengers`, `hour`,
`luggage_bags`, `fuel_price_rs`, `booking`, `zone`. The client's two rules live
in the pydantic schema, so a bad request gets a 422 and never reaches the model:
`fuel_price_rs` above 150 and `distance_km` below 0. `GET /health` returns
`{"status": "ok", "model_loaded": true}`.

## Container and cluster

`Dockerfile` builds `mp289-api:1.0` from `python:3.12-slim`; `k8s/deployment.yaml`
runs **3** replicas of `mp289-api` with readiness and liveness probes on
`/health` (liveness waits longer), and `k8s/service.yaml` exposes `mp289-svc` on
port 80 → 8000. CI (`.github/workflows/ci.yml`) runs `pytest` on every push and
then builds the image.