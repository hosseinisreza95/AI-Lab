"""Model validation: population stability and score drift.

Discrimination and calibration say whether a model was right on the data it was
tested against. Neither says whether the population it is scoring today still
looks like the one it was built on, and that is the failure mode that actually
takes credit models down: nothing errors, the AUC is never recomputed, and the
score distribution has quietly moved.

The Population Stability Index is the standard instrument:

    PSI = sum over bins of (actual_share - expected_share) * ln(actual/expected)

with the conventional reading that below 0.10 is stable, 0.10 to 0.25 warrants
investigation, and above 0.25 means the model is scoring a different population
than it was built for. The same calculation applied per feature is characteristic
drift, which is how you find out *which* input moved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

PSI_BANDS = [(0.10, "stable"), (0.25, "investigate"), (np.inf, "unstable")]
SMOOTHING = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """PSI between a reference and a current distribution.

    Bin edges come from the *expected* (reference) distribution only. Re-binning
    on the combined data would move the edges to absorb the drift, and PSI would
    report stability precisely when the population has shifted.
    """
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_share = expected_counts / max(expected_counts.sum(), 1) + SMOOTHING
    actual_share = actual_counts / max(actual_counts.sum(), 1) + SMOOTHING

    contributions = (actual_share - expected_share) * np.log(actual_share / expected_share)

    table = pd.DataFrame({
        "bin": [f"[{edges[i]:.4g}, {edges[i + 1]:.4g})" for i in range(len(edges) - 1)],
        "expected_pct": (expected_share * 100).round(2),
        "actual_pct": (actual_share * 100).round(2),
        "contribution": contributions.round(5),
    })
    return float(contributions.sum()), table


def band(index: float) -> str:
    return next(label for threshold, label in PSI_BANDS if index < threshold)


def run(save: bool = True) -> dict:
    from features.engineering import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        TARGET,
        load_applicants,
        time_split,
    )

    applicants = load_applicants()
    train_frame, test_frame = time_split(applicants)

    from models.credit_scoring import load_scorecard

    scorecard = load_scorecard()

    train_scores = scorecard.score(train_frame)
    test_scores = scorecard.score(test_frame)

    score_psi, score_table = psi(train_scores, test_scores)

    feature_drift = []
    for feature in NUMERIC_FEATURES:
        index, _ = psi(
            train_frame[feature].to_numpy(dtype=float),
            test_frame[feature].to_numpy(dtype=float),
        )
        feature_drift.append({
            "feature": feature, "psi": round(index, 5), "assessment": band(index)
        })

    for feature in CATEGORICAL_FEATURES:
        # For categoricals, compare share per level directly rather than binning.
        expected = train_frame[feature].value_counts(normalize=True)
        actual = test_frame[feature].value_counts(normalize=True)
        levels = expected.index.union(actual.index)
        expected_share = expected.reindex(levels).fillna(0).to_numpy() + SMOOTHING
        actual_share = actual.reindex(levels).fillna(0).to_numpy() + SMOOTHING
        index = float(((actual_share - expected_share) * np.log(actual_share / expected_share)).sum())
        feature_drift.append({
            "feature": feature, "psi": round(index, 5), "assessment": band(index)
        })

    feature_drift.sort(key=lambda row: row["psi"], reverse=True)

    report = {
        "reference_period": [str(train_frame["application_date"].min().date()),
                            str(train_frame["application_date"].max().date())],
        "current_period": [str(test_frame["application_date"].min().date()),
                           str(test_frame["application_date"].max().date())],
        "score_psi": round(score_psi, 5),
        "score_assessment": band(score_psi),
        "score_distribution_shift": {
            "reference_median_points": int(np.median(train_scores)),
            "current_median_points": int(np.median(test_scores)),
            "shift_points": int(np.median(test_scores) - np.median(train_scores)),
        },
        "observed_default_rate": {
            "reference_pct": round(float(train_frame[TARGET].mean()) * 100, 2),
            "current_pct": round(float(test_frame[TARGET].mean()) * 100, 2),
        },
        "score_psi_detail": score_table.to_dict(orient="records"),
        "feature_drift": feature_drift,
        "interpretation": (
            "PSI below 0.10 is stable, 0.10-0.25 warrants investigation, above "
            "0.25 means the model is scoring a materially different population "
            "than it was built on. Feature-level PSI localises the cause."
        ),
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "stability_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    return report


if __name__ == "__main__":
    report = run()

    print("Population stability")
    print(f"  Reference : {report['reference_period'][0]} to {report['reference_period'][1]}")
    print(f"  Current   : {report['current_period'][0]} to {report['current_period'][1]}")
    print(f"  Score PSI : {report['score_psi']} ({report['score_assessment']})")

    shift = report["score_distribution_shift"]
    print(f"  Median score {shift['reference_median_points']} -> "
          f"{shift['current_median_points']} ({shift['shift_points']:+d} points)")

    rate = report["observed_default_rate"]
    print(f"  Observed default rate {rate['reference_pct']}% -> {rate['current_pct']}%")

    print("\nFeature drift (highest first)")
    for row in report["feature_drift"][:8]:
        print(f"  {row['feature']:<32} PSI {row['psi']:<10} {row['assessment']}")
