import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from shipment_prep import FEATURES, build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


INTEGER_METRICS = {
    "raw_rows",
    "duplicate_rows_removed",
    "rows_after_dedup",
    "missing_distance_km_before",
    "missing_parcel_weight_before",
    "sentinel_weight_rows",
    "missing_declared_value_before",
    "missing_payment_mode_before",
    "service_type_spellings_before",
    "service_type_spellings_after",
    "rows_after_hub_merge",
    "nulls_in_features_and_target",
    "delayed_count",
    "hours_outlier_count",
}


def format_value(metric, value):
    if metric in INTEGER_METRICS:
        return int(value)
    return f"{float(value):.4f}"


raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

raw_rows = len(raw)
deduped = raw.drop_duplicates()
duplicate_rows_removed = raw_rows - len(deduped)
rows_after_dedup = len(deduped)
missing_distance_km_before = int(deduped["distance_km"].isna().sum())
missing_parcel_weight_before = int(deduped["parcel_weight_kg"].isna().sum())
sentinel_weight_rows = int((deduped["parcel_weight_kg"] > 100).sum())
missing_declared_value_before = int(deduped["declared_value"].isna().sum())
missing_payment_mode_before = int(deduped["payment_mode"].isna().sum())
service_type_spellings_before = int(deduped["service_type"].nunique())
service_type_spellings_after = int(
    deduped["service_type"].astype(str).str.strip().str.title().nunique()
)

clean_df, medians = clean_shipments(raw)
model_df = build_features(clean_df, hubs)
model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)
model_df.to_csv("data/modelling_table.csv", index=False)

rows_after_hub_merge = len(model_df)
nulls_in_features_and_target = int(
    model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
)
delayed_count = int(model_df["is_delayed"].sum())
delay_rate = delayed_count / rows_after_hub_merge

hours = model_df["delivery_hours"]
hours_q1 = float(hours.quantile(0.25))
hours_q3 = float(hours.quantile(0.75))
iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * iqr
hours_upper_bound = hours_q3 + 1.5 * iqr
hours_outlier_count = int(((hours < hours_lower_bound) | (hours > hours_upper_bound)).sum())

declared_value_mean_inr = float(model_df["declared_value_inr"].mean())
declared_value_median_inr = float(model_df["declared_value_inr"].median())
log_declared_value_mean = float(model_df["log_declared_value"].mean())
log_declared_value_median = float(model_df["log_declared_value"].median())

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", rows_after_dedup),
    ("missing_distance_km_before", missing_distance_km_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("missing_declared_value_before", missing_declared_value_before),
    ("missing_payment_mode_before", missing_payment_mode_before),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("delayed_count", delayed_count),
    ("delay_rate", delay_rate),
    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),
    ("declared_value_mean_inr", declared_value_mean_inr),
    ("declared_value_median_inr", declared_value_median_inr),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median),
]
pd.DataFrame(
    [{"metric": metric, "value": format_value(metric, value)} for metric, value in checks]
).to_csv("outputs/part_a_checks.csv", index=False)

distance_band = pd.cut(
    model_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)
group_frames = []
for dimension, series in [
    ("service_type", model_df["service_type"]),
    ("hub_tier", model_df["hub_tier"]),
    ("distance_band", distance_band),
]:
    grouped = model_df.groupby(series, observed=True)
    table = grouped.agg(
        shipments=("shipment_id", "size"),
        delayed=("is_delayed", "sum"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()
    table = table.rename(columns={table.columns[0]: "group"})
    table["dimension"] = dimension
    table["delay_rate"] = table["delayed"] / table["shipments"]
    table["delayed"] = table["delayed"].astype(int)
    table["shipments"] = table["shipments"].astype(int)
    table["mean_hours"] = table["mean_hours"].map(lambda v: f"{float(v):.4f}")
    table["delay_rate"] = table["delay_rate"].map(lambda v: f"{float(v):.4f}")
    group_frames.append(table[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]])

delay_by_group = pd.concat(group_frames, ignore_index=True)
delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

numeric = model_df.select_dtypes(include=[np.number])
corr = numeric.corr()
corr.index.name = "column"
corr.to_csv("outputs/part_a_correlations.csv", index=True, float_format="%.4f")

print("Part A complete", rows_after_hub_merge, delayed_count)
