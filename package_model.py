import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model_kwargs = {
    "random_state": 42,
    "n_estimators": n_estimators,
    "min_samples_leaf": min_samples_leaf,
}
if max_depth is not None:
    model_kwargs["max_depth"] = max_depth

pipeline = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(**model_kwargs)),
    ]
)
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame(
    {"column": list(medians.keys()), "median": [round(float(v), 4) for v in medians.values()]}
).to_csv("models/impute_values.csv", index=False)

reloaded = joblib.load("models/delay_model.pkl")
prob_mem = pipeline.predict_proba(X_test)[:, 1]
prob_rel = reloaded.predict_proba(X_test)[:, 1]
diff = float(np.max(np.abs(prob_mem - prob_rel)))
pd.DataFrame(
    [
        {"metric": "test_rows_scored", "value": int(len(X_test))},
        {"metric": "reload_max_abs_probability_diff", "value": round(diff, 4) if diff >= 1e-4 else diff},
        {"metric": "threshold", "value": threshold},
    ]
).to_csv("outputs/part_d_reload_check.csv", index=False)
print("packaged", len(X_test), diff, threshold)
