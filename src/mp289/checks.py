"""The seven kinds of data problem (STUDENT_HANDBOOK section 7).

This is the code behind the data-issues part of reports/findings.yaml. Run it
from the repository root to print the evidence again:

    python -m mp289.checks

Importing this module reads no files, so CI (which has no CSVs) can import it.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mp289.data import ROOT, load_params

SENTINELS = [-1, -999, 999, 9999]
TEXT_COLUMNS = ["booking", "zone"]


def sha256(path) -> str:
    """SHA-256 of the original file, as MLflow logs it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw" / name)


def leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, threshold: float = 1000):
    """Filled in only after the ride: missing from the holdout, copies the target."""
    missing = sorted(set(train.columns) - set(holdout.columns))
    if not missing:
        return {"present": False, "column": None}
    column = missing[0]
    trusted = train[train["fare_rs"] <= threshold]  # the typos would hide the leak
    return {
        "present": True,
        "column": column,
        "absent_from_holdout": missing,
        "corr_with_target": round(float(trusted[column].corr(trusted["fare_rs"])), 4),
        "rows_within_5_rupees": int(((train[column] - train["fare_rs"]).abs() <= 5).sum()),
    }


def unit_mixup(train: pd.DataFrame, unit_fix: dict):
    """One value of the group column recorded the column in another unit."""
    column, group_value, factor = unit_fix["column"], unit_fix["zone"], unit_fix["factor"]
    stats = train.groupby("zone")[column].agg(["count", "min", "mean", "max"])
    others = stats["mean"].drop(index=group_value).mean()
    ratio = float(stats.loc[group_value, "mean"] / others)
    return {
        "present": bool(ratio >= 10),  # an order of magnitude is a unit, not noise
        "column": column,
        "group_value": group_value,
        "factor": factor,
        "ratio_to_other_groups": round(ratio, 1),
        "by_group": {
            str(z): {k: round(float(v), 3) for k, v in row.items()}
            for z, row in stats.iterrows()
        },
    }


def sentinel_value(train: pd.DataFrame, candidates=SENTINELS):
    """A special number typed instead of a blank reading."""
    num = train.select_dtypes("number")
    hits = {
        f"{c}={v}": int((num[c] == v).sum())
        for c in num.columns for v in candidates if (num[c] == v).any()
    }
    top = {
        c: {str(k): int(v) for k, v in train[c].value_counts().head(3).items()}
        for c in num.columns
    }
    negatives = {c: int((num[c] < 0).sum()) for c in num.columns if (num[c] < 0).any()}
    return {
        "present": bool(hits),
        "column": next(iter(hits), None),
        "value": next((int(k.split("=")[1]) for k in hits), None),
        "count": sum(hits.values()),
        "sentinel_hits": hits,
        "negative_values": negatives,
        "top_values": top,
    }


def duplicate_rows(train: pd.DataFrame):
    """The same record appearing twice."""
    count = int(train.duplicated().sum())
    return {
        "present": count > 0,
        "count": count,
        "duplicated_record_ids": int(train["record_id"].duplicated().sum()),
    }


def stuck_sensor(train: pd.DataFrame, min_days: int = 7):
    """A column frozen on one value for a long run of days."""
    d = train.sort_values(["date", "record_id"], kind="stable").reset_index(drop=True)
    per_column = []
    for column in train.select_dtypes("number").columns:
        runs = (d[column] != d[column].shift()).cumsum()
        g = d.groupby(runs).agg(value=(column, "first"), rows=(column, "size"),
                                start=("date", "first"), end=("date", "last"))
        g["days"] = (pd.to_datetime(g["end"]) - pd.to_datetime(g["start"])).dt.days
        longest = g.loc[g["days"].idxmax()]
        per_column.append({
            "column": column,
            "value": float(longest["value"]),
            "rows": int(longest["rows"]),
            "days": int(longest["days"]),
            "start": str(longest["start"]),
            "end": str(longest["end"]),
        })
    worst = max(per_column, key=lambda r: (r["days"], r["rows"]))
    present = worst["days"] >= min_days
    return {
        "present": present,
        "column": worst["column"] if present else None,
        "start_date": worst["start"] if present else None,
        "end_date": worst["end"] if present else None,
        "longest_run": worst,
        "per_column": per_column,
    }


def target_typos(train: pd.DataFrame, threshold: float = 1000):
    """A few targets typed with an extra zero: ten times too big."""
    values = np.sort(train["fare_rs"].to_numpy())
    gaps = np.diff(values)
    jump = int(np.argmax(gaps))
    typos = train.loc[train["fare_rs"] > threshold, "fare_rs"].sort_values()
    return {
        "present": len(typos) > 0,
        "count": int(len(typos)),
        "values": [float(v) for v in typos],
        "fixed_values": [round(float(v), 1) for v in typos / 10],
        "largest_gap": {
            "below": float(values[jump]),
            "above": float(values[jump + 1]),
            "size": float(gaps[jump]),
        },
    }


def spelling_variants(train: pd.DataFrame, later: pd.DataFrame, columns=TEXT_COLUMNS):
    """One text column with the same category written several ways."""
    detail = {
        c: {
            "raw_levels": int(train[c].nunique()),
            "true_levels": int(later[c].nunique()),
            "variants": {str(k): int(v) for k, v in train[c].value_counts().items()},
        }
        for c in columns
    }
    messy = {c: d for c, d in detail.items() if d["raw_levels"] > d["true_levels"]}
    first = next(iter(messy.values()), None)
    return {
        "present": bool(messy),
        "column": next(iter(messy), None),
        "true_levels": first["true_levels"] if first else None,
        "by_column": detail,
    }


def main():
    p = load_params()
    train, later = load_raw("train.csv"), load_raw("later.csv")
    holdout = load_raw("holdout_features.csv")

    findings = {
        "data_sha256": sha256(ROOT / p["data"]["train"]),
        "leaking_column": leaking_column(train, holdout, p["typo_threshold"]),
        "unit_mixup": unit_mixup(train, p["unit_fix"]),
        "sentinel_value": sentinel_value(train),
        "duplicate_rows": duplicate_rows(train),
        "stuck_sensor": stuck_sensor(train),
        "target_typos": target_typos(train, p["typo_threshold"]),
        "spelling_variants": spelling_variants(train, later),
    }
    keep = ("present", "column", "count", "value", "factor", "group_value",
            "true_levels", "start_date", "end_date", "corr_with_target")
    short = {
        k: {kk: vv for kk, vv in v.items() if kk in keep}
        for k, v in findings.items() if isinstance(v, dict)
    }
    print(yaml.safe_dump({"data_sha256": findings["data_sha256"], "data_issues": short},
                         sort_keys=False, allow_unicode=True))
    return findings


if __name__ == "__main__":
    main()