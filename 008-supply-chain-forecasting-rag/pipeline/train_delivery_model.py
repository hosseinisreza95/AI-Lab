"""Train and evaluate the delivery-time model.

Evaluated on a time-based split, never a random one. A random split over shipment
data puts shipments from the same week on both sides of the split, so the model
is scored on a world where it already knows how that week went. The split here is
chronological, which is the only arrangement that resembles how the model is used.

Two baselines are reported alongside:
  - the planning system's existing `planned_transit_days` rule, which is what the
    model has to beat to be worth deploying;
  - the lane historical median, which is what a good analyst with a spreadsheet
    would produce.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from pipeline.features import (
    CATEGORICAL_COLUMNS,
    CURATED_DIR,
    FEATURE_COLUMNS,
    TARGET,
    build_features,
    write_curated,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
TEST_FRACTION = 0.22


@dataclass
class Metrics:
    mae: float
    rmse: float
    within_half_day_pct: float
    late_recall_pct: float

    def to_dict(self) -> dict:
        return {key: round(value, 4) for key, value in asdict(self).items()}


def _metrics(actual: np.ndarray, predicted: np.ndarray,
             promised: np.ndarray | None = None) -> Metrics:
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    within_half = float(np.mean(np.abs(actual - predicted) <= 0.5) * 100)

    # The operationally useful question is not the average error, it is whether
    # the model flags the shipments that will miss their promise.
    late_recall = float("nan")
    if promised is not None:
        was_late = actual > promised
        if was_late.any():
            predicted_late = predicted > promised
            late_recall = float(np.mean(predicted_late[was_late]) * 100)

    return Metrics(mae, rmse, within_half, late_recall)


def _encode(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train[CATEGORICAL_COLUMNS].astype(str))

    def to_matrix(frame: pd.DataFrame) -> np.ndarray:
        numeric = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        categorical = encoder.transform(frame[CATEGORICAL_COLUMNS].astype(str))
        return np.hstack([numeric, categorical])

    return to_matrix(train), to_matrix(test), FEATURE_COLUMNS + CATEGORICAL_COLUMNS


def train(features: pd.DataFrame | None = None, save: bool = True) -> dict:
    if features is None:
        path = CURATED_DIR / "shipment_features.parquet"
        if not path.exists():
            path = write_curated(build_features())
        features = pd.read_parquet(path)

    usable = features[~features["in_warmup"]].dropna(subset=[TARGET]).copy()
    usable = usable.sort_values("dispatch_date").reset_index(drop=True)

    split_at = int(len(usable) * (1 - TEST_FRACTION))
    train_frame, test_frame = usable.iloc[:split_at], usable.iloc[split_at:]

    x_train, x_test, feature_names = _encode(train_frame, test_frame)
    y_train = train_frame[TARGET].to_numpy(dtype=float)
    y_test = test_frame[TARGET].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.06,
        max_depth=7,
        min_samples_leaf=40,
        l2_regularization=0.8,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    promised = test_frame["planned_transit_days"].to_numpy(dtype=float)
    lane_median = test_frame["lane_hist_median_transit"].fillna(
        train_frame[TARGET].median()
    ).to_numpy(dtype=float)

    results = {
        "rows_total": int(len(features)),
        "rows_usable": int(len(usable)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_period": [
            str(train_frame["dispatch_date"].min().date()),
            str(train_frame["dispatch_date"].max().date()),
        ],
        "test_period": [
            str(test_frame["dispatch_date"].min().date()),
            str(test_frame["dispatch_date"].max().date()),
        ],
        "model": _metrics(y_test, predicted, promised).to_dict(),
        "baseline_planning_rule": _metrics(y_test, promised, promised).to_dict(),
        "baseline_lane_median": _metrics(y_test, lane_median, promised).to_dict(),
    }

    importance = permutation_importance(
        model, x_test, y_test, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error"
    )
    # With neg_mean_absolute_error, importances_mean is already
    # (MAE when shuffled - MAE baseline), so a larger value means a more
    # important feature. Negating it here would report every feature as harmful.
    results["top_features"] = [
        {
            "feature": feature_names[index],
            "mae_increase_when_shuffled": round(float(importance.importances_mean[index]), 4),
        }
        for index in np.argsort(importance.importances_mean)[::-1][:8]
    ]

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(
            {"model": model, "feature_names": feature_names,
             "categorical_columns": CATEGORICAL_COLUMNS, "feature_columns": FEATURE_COLUMNS},
            MODEL_DIR / "delivery_time_model.joblib",
        )
        (MODEL_DIR / "delivery_time_metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    return results


if __name__ == "__main__":
    print("Training delivery-time model ...")
    outcome = train()
    print(json.dumps(outcome, indent=2))

    model_mae = outcome["model"]["mae"]
    rule_mae = outcome["baseline_planning_rule"]["mae"]
    print(
        f"\nModel MAE {model_mae:.3f} days vs planning rule {rule_mae:.3f} days "
        f"({(1 - model_mae / rule_mae) * 100:.1f}% better)"
    )
