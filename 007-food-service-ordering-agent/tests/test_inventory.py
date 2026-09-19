"""Tests for the forecast-to-stock join and the closure calendar.

This is the arithmetic a manager acts on: if the projection is wrong, the agent
confidently orders the wrong quantity from the right supplier. None of these
tests need an API key or a database.

Run with:  pytest -q
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

from database.seed import WEEKDAY_FACTOR, covers_for, is_closed
from forecasting.inventory import _severity, project_product
from forecasting.model import ForecastPoint


@dataclass
class FakeProduct:
    consumption_per_100_covers: float
    lead_time_days: int = 2
    units_per_case: int = 1
    min_order_quantity: int = 1
    shelf_life_days: int = 30
    case_price_eur: float = 1.0
    unit: str = "kg"


@dataclass
class FakeStock:
    quantity_on_hand: float
    par_level: float = 100.0
    on_order: float = 0.0


def make_forecast(values: list[tuple[int, bool]], start: dt.date | None = None):
    start = start or dt.date(2026, 6, 1)
    return [
        ForecastPoint(
            service_date=start + dt.timedelta(days=index),
            predicted_covers=covers,
            lower=covers,
            upper=covers,
            is_closed=closed,
        )
        for index, (covers, closed) in enumerate(values)
    ]


def test_projection_drains_stock_at_the_consumption_rate():
    product = FakeProduct(consumption_per_100_covers=10.0)   # 10 units per 100 covers
    stock = FakeStock(quantity_on_hand=100.0)
    forecast = make_forecast([(500, False), (500, False), (500, False)])

    projections = project_product(product, stock, forecast)

    assert [p.consumption for p in projections] == [50.0, 50.0, 50.0]
    assert [p.closing_stock for p in projections] == [50.0, 0.0, -50.0]


def test_closed_days_consume_nothing():
    product = FakeProduct(consumption_per_100_covers=10.0)
    stock = FakeStock(quantity_on_hand=100.0)
    # Thursday, Friday, then a closed weekend, then Monday.
    forecast = make_forecast([(500, False), (0, True), (0, True), (500, False)])

    projections = project_product(product, stock, forecast)

    assert projections[1].consumption == 0.0
    assert projections[2].consumption == 0.0
    # Stock must be unchanged across the closure, or every weekend produces a
    # phantom stock-out on Monday morning.
    assert projections[1].closing_stock == projections[2].closing_stock == 50.0
    assert projections[3].closing_stock == 0.0


def test_stock_already_on_order_counts_towards_cover():
    product = FakeProduct(consumption_per_100_covers=10.0)
    forecast = make_forecast([(500, False), (500, False)])

    without = project_product(product, FakeStock(quantity_on_hand=60.0), forecast)
    with_order = project_product(
        product, FakeStock(quantity_on_hand=60.0, on_order=50.0), forecast
    )

    # Ignoring on_order is how a system orders the same thing twice.
    assert without[-1].closing_stock == -40.0
    assert with_order[-1].closing_stock == 10.0


def test_stockout_date_is_the_first_day_stock_goes_non_positive():
    product = FakeProduct(consumption_per_100_covers=10.0)
    stock = FakeStock(quantity_on_hand=75.0)
    forecast = make_forecast([(500, False), (500, False), (500, False)])

    projections = project_product(product, stock, forecast)
    stockout = next(p.service_date for p in projections if p.closing_stock <= 0)

    assert stockout == dt.date(2026, 6, 2)


def test_severity_escalates_when_the_order_by_date_has_passed():
    # The order-by date matters more than the stock-out date: a product with six
    # days of cover and a seven-day lead time is already too late.
    assert _severity(-1, 6.0) == "critical"
    assert _severity(0, 6.0) == "critical"
    assert _severity(2, 3.0) == "high"
    assert _severity(5, 9.0) == "medium"


def test_closure_calendar_marks_weekends_and_holidays():
    assert is_closed(dt.date(2026, 6, 6))[0] is True        # Saturday
    assert is_closed(dt.date(2026, 6, 7))[0] is True        # Sunday
    assert is_closed(dt.date(2026, 7, 14)) == (True, "public holiday")
    assert is_closed(dt.date(2026, 8, 10)) == (True, "summer shutdown")
    assert is_closed(dt.date(2026, 6, 4)) == (False, None)  # Thursday


def test_generated_covers_follow_the_weekday_shape():
    import random

    rng = random.Random(1)
    monday = dt.date(2026, 6, 1)
    thursday = dt.date(2026, 6, 4)
    friday = dt.date(2026, 6, 5)

    # Average out the noise; the point is the shape, not any single day.
    def mean_for(day: dt.date) -> float:
        return sum(covers_for(day, 400, rng) for _ in range(300)) / 300

    monday_mean, thursday_mean, friday_mean = (mean_for(d) for d in (monday, thursday, friday))

    assert friday_mean < monday_mean < thursday_mean
    assert WEEKDAY_FACTOR[5] == WEEKDAY_FACTOR[6] == 0.0
    # Friday is the remote-working dip the forecast has to learn.
    assert friday_mean / thursday_mean == pytest.approx(
        WEEKDAY_FACTOR[4] / WEEKDAY_FACTOR[3], rel=0.05
    )
