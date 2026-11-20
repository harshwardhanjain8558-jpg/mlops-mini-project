import hashlib
import pandas as pd

TRAIN = "data/raw/train.csv"
HOLD = "data/raw/holdout_features.csv"
df = pd.read_csv(TRAIN)
hold = pd.read_csv(HOLD)

print("sha256:", hashlib.sha256(open(TRAIN, "rb").read()).hexdigest())
print(df.shape, "\n", df.dtypes, "\n", df.describe(include="all").T)

# 1 leaking_column: not in holdout, and ~equal to target
print("\nNot in holdout:", set(df.columns) - set(hold.columns))
num = df.select_dtypes("number")
print(num.corrwith(df["fare_rs"]).sort_values())
print("mean |receipt - fare|:", (df["fare_paid_receipt_rs"] - df["fare_rs"]).abs().mean())

# 2 unit_mixup: per-zone stats for each numeric col
for c in ["distance_km", "wait_min", "fuel_price_rs", "luggage_bags", "passengers"]:
    print(c, "\n", df.groupby("zone")[c].agg(["min", "mean", "max"]))
# also try spelling-cleaned zone first (see 7)

# 3 sentinel_value
for c in num.columns:
    vc = df[c].value_counts()
    print(c, vc.head(3).to_dict())
for v in [-1, -999, 999, 9999]:
    print(v, (num == v).sum()[lambda s: s > 0].to_dict())

# 4 duplicate_rows
print("dups:", df.duplicated().sum())

# 5 stuck_sensor: longest run of identical values by date
d = df.sort_values("date").reset_index(drop=True)
for c in ["distance_km", "wait_min", "fuel_price_rs"]:
    run = (d[c] != d[c].shift()).cumsum()
    g = d.groupby(run).agg(val=(c, "first"), n=(c, "size"),
                           start=("date", "first"), end=("date", "last"))
    print(c, "\n", g.sort_values("n", ascending=False).head(3))

# 6 target_typos
s = df["fare_rs"].sort_values()
print(s.tail(30).values)
print("gaps:", s.diff().sort_values(ascending=False).head(5))
# count = rows far above others; verify: value/10 falls in normal range
thr = df["fare_rs"].median() * 4   # adjust after looking at the gap
print("typos:", (df["fare_rs"] > thr).sum())

# 7 spelling_variants
for c in df.select_dtypes(["object", "string"]).columns:
    if c not in ("record_id", "date"):
        print(c, df[c].value_counts(dropna=False).to_dict())
# duplicates ignoring record_id
print("dups w/o id:", df.drop(columns="record_id").duplicated().sum())

# sentinel: other candidate values
for c in num.columns:
    print(c, "zeros:", (df[c] == 0).sum(), "min:", df[c].min(), "max:", df[c].max())

# stuck sensor on every numeric column, stable sort
d = df.sort_values(["date", "record_id"], kind="stable").reset_index(drop=True)
for c in num.columns:
    run = (d[c] != d[c].shift()).cumsum()
    g = d.groupby(run).agg(val=(c, "first"), n=(c, "size"),
                           start=("date", "first"), end=("date", "last"))
    top = g.sort_values("n", ascending=False).iloc[0]
    print(c, top.to_dict())

# typo sanity: value/10 should look normal
print((df.loc[df.fare_rs > 1000, "fare_rs"] / 10).values)
        