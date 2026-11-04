import os
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from shipment_prep import build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = {row["column"]: float(row["median"]) for _, row in impute.iterrows()}
segments = pd.read_csv("outputs/hub_segments.csv")

clean_df, _ = clean_shipments(raw, medians=medians)
featured = build_features(clean_df, hubs)
X = featured[feature_order]
proba = model.predict_proba(X)[:, 1].round(4)
flagged = (proba >= threshold).astype(int)

actions = featured[["shipment_id", "hub_id"]].copy()
actions["delay_probability"] = proba
actions["flagged"] = flagged
actions = actions.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")


def action_for(row):
    if row["flagged"] == 0:
        return "No action"
    if row["segment"] == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call"


actions["action"] = actions.apply(action_for, axis=1)
actions = actions[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
actions.to_csv("outputs/new_shipment_actions.csv", index=False)
print("scored", len(actions), int(actions["flagged"].sum()))
