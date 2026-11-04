import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)


def round_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return value
    return round(float(value), 4)


def class_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round_value(accuracy),
        "precision": round_value(precision),
        "recall": round_value(recall),
        "f1": round_value(f1),
    }


df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model_rows = []
blind_pred = np.zeros(len(y_val), dtype=int)
row = class_metrics(y_val, blind_pred)
row["model"] = "blind_all_on_time"
row["precision"] = 0
model_rows.append(row)

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
logreg.fit(X_train, y_train)
row = class_metrics(y_val, logreg.predict(X_val))
row["model"] = "logistic_regression"
model_rows.append(row)

best_tree_depth = None
best_tree_f1 = -1
best_tree = None
for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    f1 = f1_score(y_val, tree.predict(X_val))
    if f1 > best_tree_f1 + 1e-12 or (abs(f1 - best_tree_f1) <= 1e-12 and (best_tree_depth is None or depth < best_tree_depth)):
        best_tree_f1 = f1
        best_tree_depth = depth
        best_tree = tree
row = class_metrics(y_val, best_tree.predict(X_val))
row["model"] = f"decision_tree_depth_{best_tree_depth}"
model_rows.append(row)

rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)
row = class_metrics(y_val, rf_default.predict(X_val))
row["model"] = "random_forest_default"
model_rows.append(row)
pd.DataFrame(model_rows)[
    ["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]
].to_csv("outputs/part_c_models.csv", index=False)

rf_for_importance = RandomForestClassifier(n_estimators=100, random_state=42)
rf_for_importance.fit(X_train, y_train)
selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)
f_scores = selector_all.scores_
ranks = pd.Series(f_scores).rank(ascending=False, method="first").astype(int)
feature_rows = []
for i, feature in enumerate(FEATURES):
    feature_rows.append(
        {
            "feature": feature,
            "f_score": round_value(f_scores[i]),
            "f_rank": int(ranks.iloc[i]),
            "rf_importance": round_value(rf_for_importance.feature_importances_[i]),
            "is_noise": feature in NOISE_COLUMNS,
        }
    )
pd.DataFrame(feature_rows).to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": round_value(f1_score(y_val, pipe.predict(X_val)))})
pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

selector = SelectKBest(f_classif, k=7)
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_sel, y_train)
rand = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)
rand.fit(X_train_sel, y_train)

default_7 = RandomForestClassifier(n_estimators=100, random_state=42)
default_7.fit(X_train_sel, y_train)


def params_to_str(params):
    return ", ".join(f"{key}={params[key]}" for key in sorted(params))


tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": params_to_str(grid.best_params_),
        "best_cv_accuracy": round_value(grid.best_score_),
        "val_f1": round_value(f1_score(y_val, grid.predict(X_val_sel))),
        "params": dict(grid.best_params_),
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": params_to_str(rand.best_params_),
        "best_cv_accuracy": round_value(rand.best_score_),
        "val_f1": round_value(f1_score(y_val, rand.predict(X_val_sel))),
        "params": dict(rand.best_params_),
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": "",
        "val_f1": round_value(f1_score(y_val, default_7.predict(X_val_sel))),
        "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    },
]
pd.DataFrame(
    [{k: v for k, v in row.items() if k != "params"} for row in tuning_rows]
).to_csv("outputs/part_c_tuning.csv", index=False)

tie_rank = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}
winner = max(tuning_rows, key=lambda row: (row["val_f1"] if row["val_f1"] != "" else -1, -tie_rank[row["search"]]))
chosen_params = dict(winner["params"])
model_kwargs = {"random_state": 42, "n_estimators": chosen_params["n_estimators"]}
if chosen_params.get("max_depth") is not None:
    model_kwargs["max_depth"] = chosen_params["max_depth"]
if "min_samples_leaf" in chosen_params:
    model_kwargs["min_samples_leaf"] = chosen_params["min_samples_leaf"]

final_model = Pipeline(
    [
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(**model_kwargs)),
    ]
)
final_model.fit(X_train, y_train)
val_proba = final_model.predict_proba(X_val)[:, 1]

threshold_rows = []
chosen_threshold = 0.1
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (val_proba >= t).astype(int)
    metrics = class_metrics(y_val, pred)
    metrics["threshold"] = t
    threshold_rows.append(metrics)
    if metrics["recall"] >= 0.80:
        chosen_threshold = t
pd.DataFrame(threshold_rows)[
    ["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]
].to_csv("outputs/part_c_thresholds.csv", index=False)

test_proba = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= chosen_threshold).astype(int)
test_metrics = class_metrics(y_test, test_pred)
test_metrics["threshold"] = chosen_threshold
pd.DataFrame([test_metrics])[
    ["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]
].to_csv("outputs/part_c_final_test.csv", index=False)

max_depth_out = "unlimited" if chosen_params.get("max_depth") is None else chosen_params["max_depth"]
pd.DataFrame(
    [
        {
            "selector_k": 7,
            "n_estimators": chosen_params["n_estimators"],
            "max_depth": max_depth_out,
            "min_samples_leaf": chosen_params.get("min_samples_leaf", 1),
            "threshold": chosen_threshold,
            "chosen_from": winner["search"],
        }
    ]
).to_csv("outputs/part_c_final_config.csv", index=False)

print("Part C complete", winner["search"], chosen_threshold, best_tree_depth)
