"""Covers forecasting.

SARIMA with a weekly seasonal period, benchmarked against a seasonal-naive
baseline on a rolling-origin backtest. The baseline matters: in a staff
restaurant "the same weekday last week" is a genuinely strong predictor, and any
model that cannot beat it is not earning its dependency.

Closed days are handled outside the model. The site is shut at weekends and on
public holidays, and feeding structural zeros into a seasonal model teaches it
that covers oscillate to zero twice a week, which corrupts the weekday shape it
is supposed to learn. The series is modelled on open days only and the calendar
puts the zeros back afterwards.
"""
from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from database.models import Covers, SessionLocal

# statsmodels is noisy about convergence on short series; the backtest reports
# the accuracy that actually matters.
warnings.filterwarnings("ignore")

SEASONAL_PERIOD = 5          # five open days a week
DEFAULT_ORDER = (1, 0, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, SEASONAL_PERIOD)


@dataclass
class ForecastPoint:
    service_date: dt.date
    predicted_covers: int
    lower: int
    upper: int
    is_closed: bool
    note: str | None = None


@dataclass
class BacktestResult:
    model_mape: float
    baseline_mape: float
    model_mae: float
    baseline_mae: float
    folds: int

    @property
    def beats_baseline(self) -> bool:
        return self.model_mape < self.baseline_mape

    def to_dict(self) -> dict:
        return {
            "model_mape_pct": round(self.model_mape, 2),
            "baseline_mape_pct": round(self.baseline_mape, 2),
            "model_mae_covers": round(self.model_mae, 1),
            "baseline_mae_covers": round(self.baseline_mae, 1),
            "folds": self.folds,
            "beats_baseline": self.beats_baseline,
        }


def load_covers(site_id: str = "SITE-01") -> pd.DataFrame:
    session = SessionLocal()
    try:
        rows = (
            session.query(Covers)
            .filter(Covers.site_id == site_id)
            .order_by(Covers.service_date)
            .all()
        )
    finally:
        session.close()

    frame = pd.DataFrame(
        [
            {
                "service_date": row.service_date,
                "covers": row.covers,
                "is_closed": bool(row.is_closed),
                "note": row.note,
            }
            for row in rows
        ]
    )
    if not frame.empty:
        frame["service_date"] = pd.to_datetime(frame["service_date"])
    return frame


def _closure_calendar(day: dt.date) -> tuple[bool, str | None]:
    """Forward-looking closures. Mirrors the rules the site actually operates on."""
    from database.seed import is_closed

    return is_closed(day)


def _seasonal_naive(open_series: pd.Series, horizon: int) -> np.ndarray:
    """Last week's same open-day, repeated. The benchmark to beat."""
    if len(open_series) < SEASONAL_PERIOD:
        return np.repeat(open_series.mean(), horizon)
    window = open_series.iloc[-SEASONAL_PERIOD:].to_numpy()
    return np.array([window[index % SEASONAL_PERIOD] for index in range(horizon)])


def _fit_sarimax(open_series: pd.Series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        open_series.to_numpy(dtype=float),
        order=DEFAULT_ORDER,
        seasonal_order=DEFAULT_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def backtest(site_id: str = "SITE-01", folds: int = 6, horizon: int = 10) -> BacktestResult:
    """Rolling-origin backtest on open days.

    Each fold trains on everything up to a cut point and predicts the next
    `horizon` open days, which is how the forecast is actually used.
    """
    frame = load_covers(site_id)
    open_series = frame.loc[~frame["is_closed"], "covers"].reset_index(drop=True)

    model_errors: list[np.ndarray] = []
    baseline_errors: list[np.ndarray] = []
    actual_values: list[np.ndarray] = []
    completed = 0

    for fold in range(folds):
        cut = len(open_series) - horizon * (folds - fold)
        if cut < SEASONAL_PERIOD * 8:
            continue

        train = open_series.iloc[:cut]
        actual = open_series.iloc[cut:cut + horizon].to_numpy(dtype=float)
        if len(actual) < horizon:
            continue

        try:
            predicted = _fit_sarimax(train).forecast(steps=horizon)
        except Exception:
            # A fold that fails to converge is skipped rather than counted as a
            # win for the baseline.
            continue

        model_errors.append(np.abs(predicted - actual))
        baseline_errors.append(np.abs(_seasonal_naive(train, horizon) - actual))
        actual_values.append(actual)
        completed += 1

    if not completed:
        return BacktestResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    model_abs = np.concatenate(model_errors)
    baseline_abs = np.concatenate(baseline_errors)
    actuals = np.concatenate(actual_values)

    return BacktestResult(
        model_mape=float(np.mean(model_abs / actuals) * 100),
        baseline_mape=float(np.mean(baseline_abs / actuals) * 100),
        model_mae=float(np.mean(model_abs)),
        baseline_mae=float(np.mean(baseline_abs)),
        folds=completed,
    )


def forecast_covers(
    horizon_days: int = 14, site_id: str = "SITE-01", confidence: float = 0.8
) -> list[ForecastPoint]:
    """Forecast the next `horizon_days` calendar days, closures included as zeros."""
    frame = load_covers(site_id)
    if frame.empty:
        raise RuntimeError("No covers history. Run `python -m database.seed` first.")

    open_series = frame.loc[~frame["is_closed"], "covers"].reset_index(drop=True)
    last_date = frame["service_date"].max().date()

    future_days = [last_date + dt.timedelta(days=offset) for offset in range(1, horizon_days + 1)]
    closures = {day: _closure_calendar(day) for day in future_days}
    open_days = [day for day in future_days if not closures[day][0]]

    if not open_days:
        return [
            ForecastPoint(day, 0, 0, 0, True, closures[day][1]) for day in future_days
        ]

    try:
        fitted = _fit_sarimax(open_series)
        prediction = fitted.get_forecast(steps=len(open_days))
        means = np.asarray(prediction.predicted_mean, dtype=float)
        interval = np.asarray(
            prediction.conf_int(alpha=1 - confidence), dtype=float
        )
        lowers, uppers = interval[:, 0], interval[:, 1]
    except Exception:
        # Degrade to the baseline rather than failing: a manager with last week's
        # numbers is better served than a manager with an error message.
        means = _seasonal_naive(open_series, len(open_days))
        spread = float(open_series.tail(60).std())
        lowers, uppers = means - 1.28 * spread, means + 1.28 * spread

    by_day = {
        day: (means[index], lowers[index], uppers[index])
        for index, day in enumerate(open_days)
    }

    points: list[ForecastPoint] = []
    for day in future_days:
        closed, note = closures[day]
        if closed:
            points.append(ForecastPoint(day, 0, 0, 0, True, note))
            continue
        mean, lower, upper = by_day[day]
        points.append(
            ForecastPoint(
                service_date=day,
                predicted_covers=max(0, int(round(mean))),
                lower=max(0, int(round(lower))),
                upper=max(0, int(round(upper))),
                is_closed=False,
            )
        )
    return points


if __name__ == "__main__":
    print("Backtest (rolling origin, 10 open days per fold):")
    print(backtest().to_dict())
    print("\nNext 14 days:")
    for point in forecast_covers(14):
        if point.is_closed:
            print(f"  {point.service_date}  closed ({point.note})")
        else:
            print(
                f"  {point.service_date}  {point.predicted_covers:>4} covers "
                f"[{point.lower}-{point.upper}]"
            )
