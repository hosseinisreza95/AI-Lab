"""Tests for the pieces where a silent error becomes a wrong number in a report.

Every test here is deterministic and needs no generated data on disk, so the
suite runs on a clean checkout.

Run with:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.engineering import WoeBinning, time_split
from models.credit_scoring import (
    BASE_ODDS,
    BASE_SCORE,
    PDO,
    ks_statistic,
    probability_to_points,
)
from models.forecasting import (
    forecast_drift,
    forecast_naive,
    forecast_seasonal_naive,
    _metrics,
)
from models.risk_analysis import herfindahl, stress_test
from validation.stability import band, psi


# ---------------------------------------------------------------------------
# scorecard scaling
# ---------------------------------------------------------------------------

def test_base_odds_map_to_the_base_score():
    # 50:1 good:bad odds must land exactly on 600 points, or every downstream
    # cut-off policy is calibrated against the wrong anchor.
    probability = 1 / (1 + BASE_ODDS)
    assert probability_to_points(np.array([probability]))[0] == BASE_SCORE


def test_doubling_the_odds_adds_pdo_points():
    low = 1 / (1 + BASE_ODDS)
    high = 1 / (1 + BASE_ODDS * 2)
    points = probability_to_points(np.array([low, high]))
    assert points[1] - points[0] == pytest.approx(PDO, abs=1)


def test_higher_points_mean_lower_risk():
    points = probability_to_points(np.array([0.02, 0.10, 0.40]))
    assert points[0] > points[1] > points[2]


def test_extreme_probabilities_do_not_produce_infinite_points():
    points = probability_to_points(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(points))


# ---------------------------------------------------------------------------
# discrimination
# ---------------------------------------------------------------------------

def test_ks_is_one_for_perfect_separation():
    actual = np.array([0, 0, 0, 1, 1, 1])
    predicted = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    assert ks_statistic(actual, predicted) == pytest.approx(1.0)


def test_ks_is_near_zero_for_a_random_score():
    rng = np.random.default_rng(3)
    actual = rng.integers(0, 2, 4000)
    predicted = rng.random(4000)
    assert ks_statistic(actual, predicted) < 0.08


# ---------------------------------------------------------------------------
# WOE binning
# ---------------------------------------------------------------------------

def make_frame(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score = rng.normal(600, 90, n)
    frame = pd.DataFrame({
        "age": rng.normal(45, 12, n),
        "annual_income": rng.lognormal(10.5, 0.5, n),
        "employment_years": rng.gamma(2.5, 2.5, n),
        "existing_loans": rng.poisson(1.2, n),
        "loan_amount": rng.lognormal(10.0, 0.6, n),
        "term_months": rng.choice([12, 24, 36, 60], n),
        "debt_to_income": rng.uniform(0.05, 1.8, n),
        "bureau_score": score,
        "months_since_last_delinquency": rng.integers(-1, 90, n),
        "segment": rng.choice(["retail", "sme", "corporate"], n),
        "sector": rng.choice(["services", "construction"], n),
        "region": rng.choice(["north", "south"], n),
        "purpose": rng.choice(["equipment", "property"], n),
        "previous_default": rng.integers(0, 2, n),
    })
    # Default depends on bureau score, so WOE must come out monotonic in it.
    frame["defaulted"] = (rng.random(n) < 1 / (1 + np.exp((score - 560) / 60))).astype(int)
    return frame


def test_woe_is_monotonic_in_a_monotonic_driver():
    frame = make_frame()
    binning = WoeBinning().fit(frame, frame["defaulted"])

    mapping = binning.woe["bureau_score"]
    # Order the bins by their lower edge, then check WOE rises with the score.
    ordered = sorted(mapping.items(), key=lambda pair: float(pair[0].split(",")[0].strip("([")))
    weights = [weight for _, weight in ordered]
    assert weights == sorted(weights), f"WOE not monotonic in bureau_score: {weights}"


def test_woe_is_finite_even_when_a_bin_has_no_defaults():
    frame = make_frame()
    # Force a clean bin: the highest scores never default.
    frame.loc[frame["bureau_score"] > frame["bureau_score"].quantile(0.9), "defaulted"] = 0

    binning = WoeBinning().fit(frame, frame["defaulted"])
    weights = list(binning.woe["bureau_score"].values())
    # Without Laplace smoothing this is +inf, and the scorecard assigns infinite
    # points to anyone in that bin.
    assert all(np.isfinite(weight) for weight in weights)


def test_unseen_categories_map_to_neutral_not_to_the_best_bin():
    frame = make_frame()
    binning = WoeBinning().fit(frame, frame["defaulted"])

    unseen = frame.head(5).copy()
    unseen["sector"] = "aerospace"          # never seen at fit time
    transformed = binning.transform(unseen)

    # 0.0 is WOE-neutral. Any other default would either reject the applicant or
    # reward them for a category the model knows nothing about.
    assert (transformed["woe_sector"] == 0.0).all()


def test_binning_transform_is_stable_across_calls():
    frame = make_frame()
    binning = WoeBinning().fit(frame, frame["defaulted"])
    first = binning.transform(frame)
    second = binning.transform(frame)
    pd.testing.assert_frame_equal(first, second)


def test_information_value_ranks_the_real_driver_first():
    frame = make_frame()
    binning = WoeBinning().fit(frame, frame["defaulted"])
    table = binning.iv_table()
    assert table.iloc[0]["feature"] == "bureau_score"


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def test_time_split_never_puts_a_later_row_in_train():
    frame = make_frame(n=500)
    frame["application_date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        np.random.default_rng(5).integers(0, 400, len(frame)), unit="D"
    )
    train, test = time_split(frame, test_fraction=0.25)

    assert len(train) + len(test) == len(frame)
    # The whole point: every training application predates every test one.
    assert train["application_date"].max() <= test["application_date"].min()


# ---------------------------------------------------------------------------
# risk
# ---------------------------------------------------------------------------

def test_herfindahl_bounds():
    # One name holding everything is maximum concentration.
    assert herfindahl(pd.Series([100.0])) == pytest.approx(1.0)
    # Four equal names give 1/4.
    assert herfindahl(pd.Series([25.0, 25.0, 25.0, 25.0])) == pytest.approx(0.25)
    # And concentration rises as the split becomes uneven.
    assert herfindahl(pd.Series([70.0, 10.0, 10.0, 10.0])) > 0.25


def make_book() -> pd.DataFrame:
    return pd.DataFrame({
        "exposure_id": ["E1", "E2", "E3"],
        "sector": ["construction", "services", "construction"],
        "probability_of_default": [0.05, 0.20, 0.10],
        "loss_given_default": [0.40, 0.60, 0.50],
        "exposure_at_default": [100_000.0, 50_000.0, 200_000.0],
        "expected_loss": [2_000.0, 6_000.0, 10_000.0],
    })


def test_stress_test_base_scenario_reproduces_the_book():
    rows = stress_test(make_book())
    base = next(row for row in rows if row["scenario"] == "base")

    # The base scenario must land on the book's own total, or every percentage
    # change reported against it is measured from the wrong origin.
    assert base["expected_loss"] == pytest.approx(18_000.0)
    assert base["change_vs_base_pct"] == pytest.approx(0.0, abs=0.01)


def test_downturn_scales_expected_loss_by_the_pd_multiplier():
    rows = stress_test(make_book())
    mild = next(row for row in rows if row["scenario"] == "mild_downturn")
    assert mild["change_vs_base_pct"] == pytest.approx(40.0, abs=0.01)


def test_sector_shock_raises_both_pd_and_lgd_for_that_sector_only():
    rows = stress_test(make_book())
    shock = next(row for row in rows if row["scenario"] == "sector_shock_construction")

    # Construction PD triples and LGD rises 25%; services is untouched. A shock
    # that moved PD alone would understate the loss, because collateral is worth
    # less precisely when it is being realised.
    construction = (0.05 * 0.40 * 1.25 * 3 * 100_000) + (0.10 * 0.50 * 1.25 * 3 * 200_000)
    services = 0.20 * 0.60 * 50_000
    assert shock["expected_loss"] == pytest.approx(construction + services, abs=1.0)


def test_loss_rate_is_expressed_against_total_exposure():
    rows = stress_test(make_book())
    base = next(row for row in rows if row["scenario"] == "base")
    assert base["loss_rate_pct"] == pytest.approx(18_000 / 350_000 * 100, abs=0.01)


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------

def test_psi_is_zero_for_an_identical_distribution():
    rng = np.random.default_rng(2)
    sample = rng.normal(0, 1, 5000)
    index, _ = psi(sample, sample)
    assert index == pytest.approx(0.0, abs=1e-6)
    assert band(index) == "stable"


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 6000)
    small, _ = psi(reference, rng.normal(0.1, 1, 6000))
    large, _ = psi(reference, rng.normal(1.2, 1, 6000))

    assert small < large
    assert band(large) == "unstable"


def test_psi_bins_come_from_the_reference_only():
    """Re-binning on the current data would absorb the drift and report calm."""
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 6000)
    shifted = rng.normal(2.0, 1, 6000)

    index, table = psi(reference, shifted)
    # A genuine 2-sigma shift must be loud, and the mass must pile into the top bin.
    assert index > 0.5
    assert table["actual_pct"].iloc[-1] > table["expected_pct"].iloc[-1] * 3


# ---------------------------------------------------------------------------
# forecasting baselines
# ---------------------------------------------------------------------------

def test_naive_repeats_the_last_observation():
    history = np.array([10.0, 11.0, 12.0])
    assert forecast_naive(history, 3).tolist() == [12.0, 12.0, 12.0]


def test_drift_extrapolates_the_average_change():
    history = np.array([10.0, 11.0, 12.0, 13.0])
    assert forecast_drift(history, 2).tolist() == [14.0, 15.0]


def test_seasonal_naive_repeats_the_last_week():
    history = np.arange(1.0, 15.0)     # 14 observations
    predicted = forecast_seasonal_naive(history, 7)
    assert predicted.tolist() == list(range(8, 15))


def test_directional_accuracy_is_not_applicable_to_a_flat_forecast():
    actual = np.array([101.0, 102.0, 103.0])
    flat = np.array([100.0, 100.0, 100.0])
    result = _metrics("naive", actual, flat, last_observed=100.0)

    # A flat forecast makes no directional call. Scoring it as 0% reads as
    # "always wrong" when the truth is "never asked".
    assert np.isnan(result.directional_accuracy)


def test_directional_accuracy_counts_only_the_steps_with_a_call():
    actual = np.array([101.0, 99.0])
    predicted = np.array([102.0, 103.0])     # up, up - second one wrong
    result = _metrics("m", actual, predicted, last_observed=100.0)
    assert result.directional_accuracy == pytest.approx(50.0)
