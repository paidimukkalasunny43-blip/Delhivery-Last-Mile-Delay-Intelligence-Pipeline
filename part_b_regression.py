import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)


def round_value(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), 4)


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())

pd.DataFrame(
    [
        {"metric": "train_rows", "value": round_value(len(X_train))},
        {"metric": "val_rows", "value": round_value(len(X_val))},
        {"metric": "test_rows", "value": round_value(len(X_test))},
        {"metric": "train_scaled_max_abs_mean", "value": round_value(train_scaled_max_abs_mean)},
        {"metric": "test_scaled_max_abs_mean", "value": round_value(test_scaled_max_abs_mean)},
    ]
).to_csv("outputs/part_b_checks.csv", index=False)

depth_rows = []
best_depth = None
best_val_rmse = np.inf
for depth in [1, 2, 3, 4, 6, 8, 12, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    _, _, train_rmse, _ = regression_metrics(y_train, train_pred)
    _, _, val_rmse, _ = regression_metrics(y_val, val_pred)
    label = "unlimited" if depth is None else depth
    depth_rows.append(
        {
            "max_depth": label,
            "train_rmse": round_value(train_rmse),
            "val_rmse": round_value(val_rmse),
        }
    )
    depth_rank = 10**9 if depth is None else depth
    best_rank = 10**9 if best_depth in (None, "unlimited") else int(best_depth)
    if best_depth is None or val_rmse < best_val_rmse - 1e-12 or (
        abs(val_rmse - best_val_rmse) <= 1e-12 and depth_rank < best_rank
    ):
        best_val_rmse = val_rmse
        best_depth = label
pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

tree_depth = None if best_depth == "unlimited" else int(best_depth)

candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=tree_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

metric_rows = []
mean_baseline = float(y_train.mean())
mae, mse, rmse, r2 = regression_metrics(y_val, np.full(len(y_val), mean_baseline))
metric_rows.append(
    {
        "model": "mean_baseline",
        "split": "val",
        "MAE": round_value(mae),
        "MSE": round_value(mse),
        "RMSE": round_value(rmse),
        "R2": round_value(r2),
    }
)

val_rmse_by_model = {}
fitted = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    mae, mse, rmse, r2 = regression_metrics(y_val, model.predict(X_val))
    val_rmse_by_model[name] = rmse
    metric_rows.append(
        {
            "model": name,
            "split": "val",
            "MAE": round_value(mae),
            "MSE": round_value(mse),
            "RMSE": round_value(rmse),
            "R2": round_value(r2),
        }
    )

lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_val)
    _, _, rmse, r2 = regression_metrics(y_val, pred)
    lr_rows.append(
        {
            "learning_rate": lr,
            "val_rmse": round_value(rmse),
            "val_r2": round_value(r2),
        }
    )
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

ranked = sorted(val_rmse_by_model.items(), key=lambda item: item[1])
top_two = [name for name, _ in ranked[:2]]
X_cv = pd.concat([X_train, X_val], axis=0)
y_cv = pd.concat([y_train, y_val], axis=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
cv_models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=tree_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name in top_two:
    scores = cross_val_score(cv_models[name], X_cv, y_cv, cv=kfold)
    cv_rows.append(
        {
            "model": name,
            "mean_r2": round_value(scores.mean()),
            "std_r2": round_value(scores.std()),
        }
    )
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranked[0][0]
final_model = fitted[final_name]
mae, mse, rmse, r2 = regression_metrics(y_test, final_model.predict(X_test))
metric_rows.append(
    {
        "model": final_name,
        "split": "test",
        "MAE": round_value(mae),
        "MSE": round_value(mse),
        "RMSE": round_value(rmse),
        "R2": round_value(r2),
    }
)
pd.DataFrame(metric_rows).to_csv("outputs/part_b_metrics.csv", index=False)
print("Part B complete", final_name, best_depth)
