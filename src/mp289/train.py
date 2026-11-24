import hashlib

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mp289.data import ROOT, load_later, load_params, load_train

CAT = ["booking", "zone"]

CONFIGS = [
    ("ridge", Ridge, {"alpha": 1.0}),
    ("ridge", Ridge, {"alpha": 10.0}),
    ("random_forest", RandomForestRegressor,
     {"n_estimators": 200, "max_depth": 8, "random_state": 42}),
    ("gradient_boosting", GradientBoostingRegressor,
     {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}),
]


def build_pipeline(model):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])


def main():
    p = load_params()
    feats, target = p["features"], p["target"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(p["mlflow"]["experiment"])

    sha = hashlib.sha256(open(ROOT / p["data"]["train"], "rb").read()).hexdigest()
    train, later = load_train(p), load_later(p)

    best = None
    for name, cls, hp in CONFIGS:
        pipe = build_pipeline(cls(**hp))
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("data_sha256", sha)
            mlflow.log_param("model", name)
            mlflow.log_params(hp)

            mae_cv = -cross_val_score(
                pipe, train[feats], train[target],
                cv=5, scoring="neg_mean_absolute_error",
            ).mean()
            pipe.fit(train[feats], train[target])
            mae_later = mean_absolute_error(later[target], pipe.predict(later[feats]))

            mlflow.log_metric("mae_cv", mae_cv)
            mlflow.log_metric("mae_later", mae_later)
            info = mlflow.sklearn.log_model(
                pipe, name="model", serialization_format="cloudpickle"
            )
            print(f"{name} {hp} cv={mae_cv:.2f} later={mae_later:.2f}")

            if best is None or mae_cv < best["mae_cv"]:
                best = dict(mae_cv=mae_cv, run_id=run.info.run_id,
                            uri=info.model_uri, pipe=pipe)

    model_name = p["mlflow"]["model_name"]
    mv = mlflow.register_model(best["uri"], model_name)
    MlflowClient().set_registered_model_alias(model_name, "champion", mv.version)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best["pipe"], ROOT / "models" / "model.joblib")

    print("\n--- for findings.yaml ---")
    print("experiment_name:", p["mlflow"]["experiment"])
    print("best_run_id:", best["run_id"])
    print("registered_model_name:", model_name)
    print("champion_version:", mv.version)


if __name__ == "__main__":
    main()