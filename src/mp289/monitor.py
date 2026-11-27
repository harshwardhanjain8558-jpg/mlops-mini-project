"""Stage 4: drift, retraining and the promotion decision (PROBLEM.md section 7)."""

import hashlib
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from mp289.data import ROOT, load_later, load_params, load_train
from mp289.predict import predict_holdout, write_predictions
from mp289.train import build_pipeline

# every numeric column the findings form asks about (fare_paid_receipt_rs is dropped
# as the leaking column, fare_rs is the target)
PSI_COLUMNS = ["distance_km", "wait_min", "passengers", "hour", "luggage_bags", "fuel_price_rs"]


def psi(expected, actual, bins=10, eps=1e-4) -> float:
    """Population Stability Index between two columns, binned on the expected one."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # constant column: one bin around the value
        edges = np.array([expected.min() - 0.5, expected.max() + 0.5])
    edges = np.array(edges, dtype=float)
    # keep values that fall outside the training range in the outer bins
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=edges)[0] / expected.size
    a = np.histogram(actual, bins=edges)[0] / actual.size
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(train: pd.DataFrame, later: pd.DataFrame, columns=PSI_COLUMNS, bins=10) -> dict:
    """PSI per numeric column, cleaned train.csv vs later.csv."""
    return {c: round(psi(train[c], later[c], bins), 4) for c in columns}


def mae_by_group(model, df: pd.DataFrame, target: str, features, group="zone") -> dict:
    """The model's MAE on each value of `group`."""
    scores = {}
    for value, part in df.groupby(group):
        mae = mean_absolute_error(part[target], model.predict(part[list(features)]))
        scores[value] = round(float(mae), 3)
    return scores


def time_split(df: pd.DataFrame, fraction=0.7, date_col="date"):
    """First `fraction` of the rows by date, then the rest. No row is in both."""
    ordered = df.sort_values([date_col, "record_id"], kind="stable").reset_index(drop=True)
    cut = int(round(len(ordered) * fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def should_promote(mae_v1: float, mae_v2: float, improvement: float = 0.12) -> bool:
    """The client's rule: v2 must be at least `improvement` lower than v1."""
    return mae_v2 <= mae_v1 * (1 - improvement)


def update_findings(path, monitoring: dict, champion_version=None) -> str:
    """Replace the monitoring block in findings.yaml, keeping everything above it."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    head = text.split("\nmonitoring:", 1)[0].rstrip() + "\n\n"
    body = yaml.safe_dump({"monitoring": monitoring}, sort_keys=False, allow_unicode=True)
    new = head + body

    if champion_version is not None:
        lines = []
        for line in new.splitlines():
            if line.strip().startswith("champion_version:"):
                indent = line[: len(line) - len(line.lstrip())]
                line = f"{indent}champion_version: {champion_version}"
            lines.append(line)
        new = "\n".join(lines) + "\n"

    path.write_text(new, encoding="utf-8")
    return new


def main(findings_path=None, predictions_path=None, model_path=None, v1_path=None) -> dict:
    p = load_params()
    feats, target = p["features"], p["target"]
    mon = p.get("monitoring", {})
    bins = mon.get("psi_bins", 10)
    fraction = mon.get("retrain_fraction", 0.7)
    improvement = mon.get("promotion_improvement", 0.12)

    findings_path = Path(findings_path or ROOT / "reports/findings.yaml")
    predictions_path = Path(predictions_path or ROOT / "predictions/holdout_predictions.csv")
    model_path = Path(model_path or ROOT / "models/model.joblib")
    v1_path = Path(v1_path or ROOT / "models/model_v1.joblib")

    # v1 is the model the service serves. Keep a copy so re-running never treats a
    # promoted v2 as the baseline.
    if not v1_path.exists():
        shutil.copyfile(model_path, v1_path)
    v1 = joblib.load(v1_path)

    train, later = load_train(p), load_later(p)
    if target not in later.columns:
        raise SystemExit(f"{p['data']['later']} has no {target} column")

    # 1. drift: PSI per numeric column, cleaned train.csv vs later.csv
    psi_values = psi_table(train, later, PSI_COLUMNS, bins)
    most_drifted = max(psi_values, key=psi_values.get)

    # 2. v1's MAE on later.csv, per zone
    by_zone = mae_by_group(v1, later, target, feats)
    worst_group = max(by_zone, key=by_zone.get)

    # 3. v2: cleaned train.csv + the first 70% of later.csv (by date)
    fit_part, check_part = time_split(later, fraction)
    mae_v1_check = float(mean_absolute_error(
        check_part[target], v1.predict(check_part[feats])
    ))

    v2 = build_pipeline(clone(v1.named_steps["model"]))
    v2.fit(
        pd.concat([train[feats], fit_part[feats]], ignore_index=True),
        pd.concat([train[target], fit_part[target]], ignore_index=True),
    )
    mae_v2_check = float(mean_absolute_error(
        check_part[target], v2.predict(check_part[feats])
    ))

    # 4. the client's promotion rule
    promote = should_promote(mae_v1_check, mae_v2_check, improvement)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(p["mlflow"]["experiment"])
    sha = hashlib.sha256(open(ROOT / p["data"]["train"], "rb").read()).hexdigest()
    model_name = p["mlflow"]["model_name"]

    with mlflow.start_run(run_name="v2_retrain") as run:
        mlflow.log_param("data_sha256", sha)
        mlflow.log_param("model", type(v1.named_steps["model"]).__name__)
        mlflow.log_param("retrain_rows", len(fit_part))
        mlflow.log_param("retrain_fraction", fraction)
        mlflow.log_param("promotion_improvement", improvement)
        mlflow.log_metric("mae_check_v1", mae_v1_check)
        mlflow.log_metric("mae_check_v2", mae_v2_check)
        info = mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
        v2_run_id = run.info.run_id

    mv = mlflow.register_model(info.model_uri, model_name)
    version = int(mv.version)

    client = MlflowClient()
    if promote:
        client.set_registered_model_alias(model_name, "champion", version)
        joblib.dump(v2, model_path)
        champion = v2
    else:
        champion = v1  # the alias still points at v1

    # 5. the final prediction, from the champion that is now serving
    hold = pd.read_csv(ROOT / p["data"]["holdout"])
    predictions = predict_holdout(champion, hold, feats)
    write_predictions(predictions, predictions_path)

    monitoring = {
        "psi": psi_values,
        "most_drifted_feature": most_drifted,
        "mae_by_group": by_zone,
        "worst_served_group": worst_group,
        "mae_v1_on_check": round(mae_v1_check, 3),
        "mae_v2_on_check": round(mae_v2_check, 3),
        "decision": "promote" if promote else "keep",
    }
    update_findings(
        findings_path,
        monitoring,
        champion_version=version if promote else int(client.get_model_version_by_alias(model_name, "champion").version),
    )

    summary = {
        "psi": psi_values,
        "most_drifted_feature": most_drifted,
        "mae_by_group": by_zone,
        "worst_served_group": worst_group,
        "mae_v1_on_check": round(mae_v1_check, 3),
        "mae_v2_on_check": round(mae_v2_check, 3),
        "improvement_pct": round((mae_v1_check - mae_v2_check) / mae_v1_check * 100, 2),
        "promotion_threshold_pct": improvement * 100,
        "decision": monitoring["decision"],
        "registered_version": version,
        "v2_run_id": v2_run_id,
        "champion_version": int(client.get_model_version_by_alias(model_name, "champion").version),
        "predictions": str(predictions_path),
        "rows": len(predictions),
    }
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    return summary


if __name__ == "__main__":
    main()
