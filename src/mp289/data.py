import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_params(path=ROOT / "params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def clean_booking(s: pd.Series) -> pd.Series:
    s = s.str.strip().str.lower().str.rstrip(".")
    return s.replace({"sta": "stand", "str": "street"})


def clean_common(df: pd.DataFrame) -> pd.DataFrame:
    """Safe for train, later and holdout."""
    df = df.copy()
    df = df.drop(columns=["fare_paid_receipt_rs"], errors="ignore")
    df["booking"] = clean_booking(df["booking"])
    return df


def clean_train(df: pd.DataFrame, params=None) -> pd.DataFrame:
    """Extra fixes that exist only in train.csv."""
    p = params or load_params()
    df = clean_common(df)
    u = p["unit_fix"]
    m = df["zone"] == u["zone"]
    df.loc[m, u["column"]] = df.loc[m, u["column"]] / u["factor"]
    t = p["target"]
    df[t] = df[t].astype(float)
    typo = df[t] > p["typo_threshold"]
    df.loc[typo, t] = df.loc[typo, t] / 10
    return df.drop_duplicates()


def load_train(params=None):
    p = params or load_params()
    return clean_train(pd.read_csv(ROOT / p["data"]["train"]), p)


def load_later(params=None):
    p = params or load_params()
    return clean_common(pd.read_csv(ROOT / p["data"]["later"]))