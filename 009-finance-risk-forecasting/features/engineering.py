"""Feature engineering for credit scoring.

Two representations of the same applicants, built from the same fit:

- **Binned / WOE features** for the scorecard. Every continuous variable is cut
  into monotonic-ish bins and replaced by the Weight of Evidence of the bin. This
  is not nostalgia: it is what lets a logistic regression represent a
  non-monotonic relationship (young and old applicants are both riskier than
  middle-aged ones) while staying a linear model a credit officer can read.
- **Raw numeric + one-hot features** for the gradient-boosting challenger, which
  needs no binning and is there to measure what the interpretability constraint
  actually costs.

Bin edges and WOE values are fitted on the training split only and applied to the
test split. Fitting them on everything is leakage - the bins would be drawn using
outcomes the model is about to be scored on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

TARGET = "defaulted"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "existing_loans",
    "loan_amount",
    "term_months",
    "debt_to_income",
    "bureau_score",
    "months_since_last_delinquency",
]
CATEGORICAL_FEATURES = ["segment", "sector", "region", "purpose"]
BINARY_FEATURES = ["previous_default"]

N_BINS = 6
# Laplace smoothing on the WOE numerator and denominator. Without it, a bin with
# zero defaults produces an infinite weight and a scorecard that assigns infinite
# points to whoever lands in it.
WOE_SMOOTHING = 0.5


@dataclass
class WoeBinning:
    """Fitted binning and WOE mapping. Fit on train, apply to anything."""

    edges: dict[str, np.ndarray] = field(default_factory=dict)
    woe: dict[str, dict] = field(default_factory=dict)
    iv: dict[str, float] = field(default_factory=dict)

    # -- fitting -----------------------------------------------------------

    def _fit_column(self, labels: pd.Series, target: pd.Series, column: str) -> None:
        total_bad = max(target.sum(), 1)
        total_good = max(len(target) - total_bad, 1)

        mapping: dict = {}
        information_value = 0.0

        for level in labels.unique():
            mask = labels == level
            bad = float(target[mask].sum())
            good = float(mask.sum() - bad)

            bad_rate = (bad + WOE_SMOOTHING) / (total_bad + WOE_SMOOTHING * labels.nunique())
            good_rate = (good + WOE_SMOOTHING) / (total_good + WOE_SMOOTHING * labels.nunique())

            weight = float(np.log(good_rate / bad_rate))
            mapping[level] = weight
            information_value += (good_rate - bad_rate) * weight

        self.woe[column] = mapping
        self.iv[column] = float(information_value)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "WoeBinning":
        for column in NUMERIC_FEATURES:
            series = frame[column]
            # Quantile bins, de-duplicated: several features are heavily skewed
            # and equal-width bins would put 95% of applicants in one bucket.
            _, edges = pd.qcut(series, N_BINS, retbins=True, duplicates="drop")
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges[column] = edges
            labels = pd.cut(series, bins=edges, include_lowest=True).astype(str)
            self._fit_column(labels, target, column)

        for column in CATEGORICAL_FEATURES + BINARY_FEATURES:
            self._fit_column(frame[column].astype(str), target, column)

        return self

    # -- applying ----------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        for column in NUMERIC_FEATURES:
            labels = pd.cut(
                frame[column], bins=self.edges[column], include_lowest=True
            ).astype(str)
            mapping = self.woe[column]
            # A level unseen at fit time maps to 0.0, which is WOE-neutral: the
            # applicant is treated as average on that variable rather than being
            # dropped or silently scored as the best bin.
            out[f"woe_{column}"] = labels.map(mapping).fillna(0.0).astype(float)

        for column in CATEGORICAL_FEATURES + BINARY_FEATURES:
            mapping = self.woe[column]
            out[f"woe_{column}"] = (
                frame[column].astype(str).map(mapping).fillna(0.0).astype(float)
            )

        return out

    def bin_label(self, column: str, value: float) -> str:
        """Which bin a value falls into. Used when explaining one decision."""
        if column not in self.edges:
            return str(value)
        return str(pd.cut([value], bins=self.edges[column], include_lowest=True)[0])

    def iv_table(self) -> pd.DataFrame:
        """Information Value per feature, with the conventional reading.

        The bands are industry convention, not statistics, and they are here
        because a credit officer will ask which variables the model leans on.
        """
        frame = pd.DataFrame(
            [{"feature": key, "information_value": round(value, 4)}
             for key, value in self.iv.items()]
        ).sort_values("information_value", ascending=False)
        frame["strength"] = pd.cut(
            frame["information_value"],
            bins=[-0.01, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["useless", "weak", "medium", "strong", "suspicious"],
        ).astype(str)
        return frame.reset_index(drop=True)


def build_raw_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw numeric plus one-hot categoricals, for the gradient-boosting challenger."""
    columns = NUMERIC_FEATURES + BINARY_FEATURES
    train_out = pd.get_dummies(
        train[columns + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES
    )
    test_out = pd.get_dummies(
        test[columns + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES
    )
    # Align so a category absent from the test split does not shift the columns.
    test_out = test_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out.astype(float), test_out.astype(float)


def load_applicants() -> pd.DataFrame:
    path = RAW_DIR / "applicants.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m data.generate` first."
        )
    frame = pd.read_parquet(path)
    frame["application_date"] = pd.to_datetime(frame["application_date"])
    return frame


def time_split(frame: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by application date, never at random.

    Credit portfolios drift: the mix of applicants, the economy and the lending
    policy all change. A random split scores the model on a population it has
    already seen, which is not the population it will meet.
    """
    ordered = frame.sort_values("application_date").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
