"""Credit scoring: a WOE scorecard, and a gradient-boosting challenger.

The scorecard is the model that would actually be deployed, because the people
who act on it have to be able to explain a decline. The challenger exists to
measure what that constraint costs in discrimination - which is a number worth
knowing before anyone argues about it.

Two things this module insists on:

1. **Calibration, not just ranking.** AUC says the model orders applicants
   correctly. It says nothing about whether a predicted 4% default probability
   means 4 defaults in 100. The risk layer multiplies PD by LGD by EAD to get an
   expected loss in euros, so an uncalibrated PD produces a confidently wrong
   number. Brier score and a decile calibration table are reported alongside.
2. **Points, not probabilities, in the output.** A scorecard converts log-odds to
   integer points with a documented PDO (points to double the odds), because that
   is the artifact a credit officer works with.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from features.engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    WoeBinning,
    build_raw_matrix,
    load_applicants,
    time_split,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Scorecard scaling. These are the conventional defaults: 600 points at 50:1
# odds, doubling every 20 points.
BASE_SCORE = 600
BASE_ODDS = 50.0
PDO = 20.0


@dataclass
class ScoreMetrics:
    auc: float
    gini: float
    ks: float
    brier: float
    default_rate: float

    def to_dict(self) -> dict:
        return {
            "auc": round(self.auc, 4),
            "gini": round(self.gini, 4),
            "ks": round(self.ks, 4),
            "brier": round(self.brier, 5),
            "default_rate_pct": round(self.default_rate * 100, 2),
        }


def ks_statistic(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Maximum separation between the cumulative good and bad distributions.

    Still the number a credit risk team asks for first, because it corresponds to
    the cut-off decision: it is the score threshold where the two populations are
    most separated.
    """
    order = np.argsort(predicted)
    labels = actual[order]
    bad = np.cumsum(labels) / max(labels.sum(), 1)
    good = np.cumsum(1 - labels) / max((1 - labels).sum(), 1)
    return float(np.max(np.abs(good - bad)))


def evaluate(actual: np.ndarray, probability: np.ndarray) -> ScoreMetrics:
    auc = float(roc_auc_score(actual, probability))
    return ScoreMetrics(
        auc=auc,
        gini=2 * auc - 1,
        ks=ks_statistic(actual, probability),
        brier=float(brier_score_loss(actual, probability)),
        default_rate=float(actual.mean()),
    )


def calibration_table(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict]:
    """Predicted vs observed default rate by decile of predicted risk.

    This is the table that shows whether a 4% prediction means 4%. Expected loss
    is computed from these probabilities, so a systematic gap here becomes a
    systematic error in euros.
    """
    frame = pd.DataFrame({"actual": actual, "probability": probability})
    frame["decile"] = pd.qcut(frame["probability"], bins, labels=False, duplicates="drop")

    grouped = frame.groupby("decile").agg(
        applicants=("actual", "size"),
        predicted_rate=("probability", "mean"),
        observed_rate=("actual", "mean"),
    ).reset_index()
    grouped["difference_pp"] = (
        (grouped["observed_rate"] - grouped["predicted_rate"]) * 100
    ).round(2)
    grouped["predicted_rate"] = (grouped["predicted_rate"] * 100).round(2)
    grouped["observed_rate"] = (grouped["observed_rate"] * 100).round(2)
    return grouped.to_dict(orient="records")


def probability_to_points(probability: np.ndarray) -> np.ndarray:
    """Convert a default probability to scorecard points.

    factor/offset are the standard scaling. Higher points mean lower risk, which
    is the direction every downstream user expects.
    """
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)

    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    odds_good = (1 - probability) / probability
    return np.round(offset + factor * np.log(odds_good)).astype(int)


@dataclass
class ScorecardModel:
    binning: WoeBinning
    model: LogisticRegression
    feature_names: list[str]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.binning.transform(frame)[self.feature_names]
        return self.model.predict_proba(matrix)[:, 1]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return probability_to_points(self.predict_proba(frame))

    def coefficient_table(self) -> pd.DataFrame:
        """The scorecard, as a credit officer reads it.

        WOE is oriented so a higher value means a better applicant. A negative
        coefficient on a WOE feature would therefore mean the model believes
        better applicants default more - a sign reversal, and always a reason to
        stop and look rather than ship.
        """
        frame = pd.DataFrame({
            "feature": [name.replace("woe_", "") for name in self.feature_names],
            "coefficient": self.model.coef_[0].round(4),
        })
        frame["direction"] = np.where(
            frame["coefficient"] > 0, "as expected", "SIGN REVERSAL - investigate"
        )
        frame["abs_coefficient"] = frame["coefficient"].abs()
        return frame.sort_values("abs_coefficient", ascending=False).drop(
            columns="abs_coefficient"
        ).reset_index(drop=True)

    def explain(self, row: pd.Series, top_n: int = 5) -> dict:
        """Why this applicant got this score.

        A declined applicant is entitled to a reason, and a credit officer needs
        one they can say out loud. Each feature's contribution is its WOE times
        its coefficient - which is exactly how the linear model reached its
        answer, not an approximation of it.
        """
        frame = row.to_frame().T
        woe = self.binning.transform(frame)[self.feature_names].iloc[0]
        contributions = woe.to_numpy() * self.model.coef_[0]

        ordered = np.argsort(contributions)[::-1]
        reasons = []
        for index in ordered[:top_n]:
            name = self.feature_names[index].replace("woe_", "")
            value = row.get(name)
            reasons.append({
                "feature": name,
                "value": value if not isinstance(value, (np.floating, np.integer)) else float(value),
                "bin": self.binning.bin_label(name, value) if name in NUMERIC_FEATURES else str(value),
                "contribution_to_risk": round(float(contributions[index]), 4),
            })

        probability = float(self.predict_proba(frame)[0])
        return {
            "probability_of_default_pct": round(probability * 100, 2),
            "score_points": int(probability_to_points(np.array([probability]))[0]),
            "top_risk_drivers": reasons,
        }


def save_scorecard(scorecard: ScorecardModel, path: Path) -> None:
    """Persist the scorecard as plain data, not as a pickled class.

    Pickling the dataclass records the class by import path, so an artifact
    written by `python -m models.credit_scoring` stores it as
    `__main__.ScorecardModel` and fails to load anywhere else. Bin edges, WOE
    maps and coefficients are all the model actually is, and they survive both
    a different entry point and a refactor of this file.
    """
    import joblib

    joblib.dump(
        {
            "edges": scorecard.binning.edges,
            "woe": scorecard.binning.woe,
            "iv": scorecard.binning.iv,
            "coefficients": scorecard.model.coef_,
            "intercept": scorecard.model.intercept_,
            "feature_names": scorecard.feature_names,
            "scaling": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO},
        },
        path,
    )


def load_scorecard(path: Path | None = None) -> ScorecardModel:
    """Rebuild a scorecard from its saved components."""
    import joblib

    path = path or OUTPUT_DIR / "scorecard.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m models.credit_scoring` first."
        )

    bundle = joblib.load(path)
    binning = WoeBinning(edges=bundle["edges"], woe=bundle["woe"], iv=bundle["iv"])

    model = LogisticRegression()
    model.coef_ = bundle["coefficients"]
    model.intercept_ = bundle["intercept"]
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = len(bundle["feature_names"])
    # Without this, scoring a named DataFrame warns that the model was fitted
    # without feature names - true of the reconstruction, misleading about the fit.
    model.feature_names_in_ = np.array(bundle["feature_names"], dtype=object)

    return ScorecardModel(binning, model, bundle["feature_names"])


def train(test_fraction: float = 0.25, save: bool = True) -> dict:
    applicants = load_applicants()
    train_frame, test_frame = time_split(applicants, test_fraction)

    y_train = train_frame[TARGET].to_numpy()
    y_test = test_frame[TARGET].to_numpy()

    # --- the scorecard ----------------------------------------------------
    binning = WoeBinning().fit(train_frame, train_frame[TARGET])
    x_train = binning.transform(train_frame)
    x_test = binning.transform(test_frame)
    feature_names = list(x_train.columns)

    logistic = LogisticRegression(max_iter=2000, C=1.0)
    logistic.fit(x_train, y_train)
    scorecard = ScorecardModel(binning, logistic, feature_names)

    scorecard_probability = logistic.predict_proba(x_test)[:, 1]
    scorecard_metrics = evaluate(y_test, scorecard_probability)

    # --- the challenger ---------------------------------------------------
    raw_train, raw_test = build_raw_matrix(train_frame, test_frame)
    booster = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        min_samples_leaf=60, l2_regularization=1.0, random_state=42,
    )
    # Boosting ranks well and is poorly calibrated out of the box. Since the risk
    # layer turns these probabilities into euros, the challenger is wrapped in
    # isotonic calibration so the comparison is like for like.
    calibrated = CalibratedClassifierCV(booster, method="isotonic", cv=3)
    calibrated.fit(raw_train, y_train)
    challenger_probability = calibrated.predict_proba(raw_test)[:, 1]
    challenger_metrics = evaluate(y_test, challenger_probability)

    results = {
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_period": [str(train_frame["application_date"].min().date()),
                         str(train_frame["application_date"].max().date())],
        "test_period": [str(test_frame["application_date"].min().date()),
                        str(test_frame["application_date"].max().date())],
        "scorecard": scorecard_metrics.to_dict(),
        "challenger_gbm": challenger_metrics.to_dict(),
        "gini_cost_of_interpretability": round(
            challenger_metrics.gini - scorecard_metrics.gini, 4
        ),
        "information_values": binning.iv_table().to_dict(orient="records"),
        "scorecard_coefficients": scorecard.coefficient_table().to_dict(orient="records"),
        "calibration_scorecard": calibration_table(y_test, scorecard_probability),
        "score_distribution": {
            "p05": int(np.percentile(probability_to_points(scorecard_probability), 5)),
            "p50": int(np.percentile(probability_to_points(scorecard_probability), 50)),
            "p95": int(np.percentile(probability_to_points(scorecard_probability), 95)),
        },
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import joblib

        save_scorecard(scorecard, OUTPUT_DIR / "scorecard.joblib")
        joblib.dump(calibrated, OUTPUT_DIR / "challenger_gbm.joblib")
        (OUTPUT_DIR / "credit_scoring_report.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        # Scored test population, consumed by the risk layer.
        scored = test_frame[["application_id", "segment", "sector", "region"]].copy()
        scored["probability_of_default"] = scorecard_probability.round(6)
        scored["score_points"] = probability_to_points(scorecard_probability)
        scored.to_parquet(OUTPUT_DIR / "scored_applicants.parquet", index=False)

    return results


if __name__ == "__main__":
    print("Training credit scoring models ...")
    report = train()

    print("\nDiscrimination and calibration on the held-out period:")
    print(f"  Scorecard (WOE + logistic) : {report['scorecard']}")
    print(f"  Challenger (calibrated GBM): {report['challenger_gbm']}")
    print(f"  Gini given up for interpretability: {report['gini_cost_of_interpretability']:+.4f}")

    print("\nTop features by Information Value:")
    for row in report["information_values"][:6]:
        print(f"  {row['feature']:<32} IV {row['information_value']:.3f}  ({row['strength']})")

    print("\nCalibration by decile (predicted vs observed default rate):")
    for row in report["calibration_scorecard"]:
        print(
            f"  decile {int(row['decile']):>2}  n={row['applicants']:>5}  "
            f"predicted {row['predicted_rate']:>6.2f}%  observed {row['observed_rate']:>6.2f}%  "
            f"diff {row['difference_pp']:+.2f}pp"
        )
