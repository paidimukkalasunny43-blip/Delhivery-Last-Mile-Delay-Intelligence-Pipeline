import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

os.makedirs("outputs", exist_ok=True)


def round_value(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), 4)


df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

profile = (
    df.groupby("hub_id", as_index=False)
    .agg(
        shipments=("shipment_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)
profile_out = profile.copy()
for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
    profile_out[col] = profile_out[col].round(4)
profile_out.to_csv("outputs/hub_profile.csv", index=False)

cluster_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
X = StandardScaler().fit_transform(profile[cluster_cols])


def pair_ratio(labels, matrix):
    across = []
    within = []
    for i, j in combinations(range(len(labels)), 2):
        dist = np.linalg.norm(matrix[i] - matrix[j])
        if labels[i] == labels[j]:
            within.append(dist)
        else:
            across.append(dist)
    return float(np.mean(across) / np.mean(within))


sep_rows = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
    sep_rows.append(
        {
            "K": k,
            "across_over_within": round_value(pair_ratio(labels, X)),
            "cluster_sizes": "-".join(str(s) for s in sizes),
        }
    )
pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

labels3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
profile["cluster"] = labels3
cluster_delay = profile.groupby("cluster")["delay_rate"].mean().sort_values()
name_map = {
    cluster_delay.index[0]: "Standard",
    cluster_delay.index[1]: "Watch",
    cluster_delay.index[2]: "Escalate",
}
profile["segment"] = profile["cluster"].map(name_map)

segments = profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/hub_segments.csv", index=False)

seg_profile = (
    profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
)
order = {"Standard": 0, "Watch": 1, "Escalate": 2}
seg_profile = seg_profile.sort_values("segment", key=lambda s: s.map(order))
for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
    seg_profile[col] = seg_profile[col].round(4)
seg_profile.to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments.copy()
mismatch["implied"] = mismatch["hub_tier"].map(implied)
mismatch = mismatch[mismatch["segment"] != mismatch["implied"]]
mismatch[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv(
    "outputs/tier_vs_segment_mismatch.csv", index=False
)
print("Part D complete", mismatch["hub_id"].tolist())
