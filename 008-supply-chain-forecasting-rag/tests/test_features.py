"""Tests for the feature layer.

The two things worth pinning down here:

1. No target leakage. A historical aggregate must never include the row it is
   attached to, or any later row. Leakage does not throw an exception, it makes
   the offline metric better, which is why it needs a test.
2. Engine parity. Spark and pandas are two implementations of one contract, and
   a dual implementation nobody checks is two implementations that have already
   diverged.

The parity test is skipped where no JVM is available, so the suite runs anywhere.

Run with:  pytest -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    TARGET,
    build_features_pandas,
    spark_available,
)


def make_shipments(rows: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.datetime(2026, 1, 1, 8, 0)
    carriers = ["CAR-A", "CAR-B"]
    stores = ["ST-1", "ST-2"]

    records = []
    for index in range(rows):
        dispatch = start + dt.timedelta(days=index // 2, hours=int(index % 2) * 6)
        transit = float(rng.uniform(0.5, 3.5))
        planned = 2
        records.append({
            "shipment_id": f"SHP-{index:05d}",
            "order_id": f"ORD-{index:05d}",
            "origin_warehouse": "WH-1",
            "destination_store": stores[index % 2],
            "region": "R1",
            "carrier": carriers[index % 2],
            "service_level": "standard",
            "category": "apparel",
            "distance_km": 100 + index,
            "units": 300,
            "weight_kg": 800.0,
            "warehouse_load": 0.8,
            "dispatch_ts": dispatch,
            "promised_ts": dispatch + dt.timedelta(days=planned),
            "actual_arrival_ts": dispatch + dt.timedelta(days=transit),
            "planned_transit_days": planned,
            "actual_transit_days": round(transit, 3),
            "status": "delivered",
        })
    return pd.DataFrame(records)


def test_first_row_of_a_group_has_no_history():
    frame = build_features_pandas(make_shipments())

    for carrier in frame["carrier"].unique():
        group = frame[frame["carrier"] == carrier].sort_values("dispatch_date")
        # The earliest shipment for a carrier cannot know that carrier's average,
        # because it is the only observation and it is its own.
        assert pd.isna(group.iloc[0]["carrier_hist_mean_transit"])


def test_historical_mean_excludes_the_row_itself():
    frame = build_features_pandas(make_shipments()).sort_values(
        ["carrier", "dispatch_date", "shipment_id"]
    )

    for carrier in frame["carrier"].unique():
        group = frame[frame["carrier"] == carrier].reset_index(drop=True)
        for position in range(1, min(len(group), 8)):
            expected = group.loc[: position - 1, TARGET].mean()
            assert group.loc[position, "carrier_hist_mean_transit"] == pytest.approx(expected)


def test_historical_features_do_not_see_the_future():
    """The strongest leakage check: truncating the dataset must not change the
    features of the rows that survive."""
    full = build_features_pandas(make_shipments(rows=60))
    truncated = build_features_pandas(make_shipments(rows=60).iloc[:30])

    merged = truncated.merge(
        full, on="shipment_id", suffixes=("_truncated", "_full")
    )
    assert len(merged) == 30

    for column in ("carrier_hist_mean_transit", "lane_hist_median_transit",
                   "warehouse_hist_mean_transit", "carrier_hist_late_rate"):
        left = merged[f"{column}_truncated"].to_numpy(dtype=float)
        right = merged[f"{column}_full"].to_numpy(dtype=float)
        # If later rows leaked in, removing them would move these values.
        assert np.allclose(left, right, equal_nan=True), f"{column} depends on future rows"


def test_lane_shipment_count_is_a_prior_count():
    frame = build_features_pandas(make_shipments()).sort_values(
        ["lane", "dispatch_date", "shipment_id"]
    )
    for lane in frame["lane"].unique():
        group = frame[frame["lane"] == lane].reset_index(drop=True)
        assert group.loc[0, "lane_hist_shipments"] == 0
        assert group.loc[1, "lane_hist_shipments"] == 1


def test_warmup_rows_are_flagged_not_filled():
    frame = build_features_pandas(make_shipments(rows=200))

    # Filling an unknown history with a global mean would put the dataset average
    # (a future-aware quantity) into the earliest rows, which is leakage wearing a
    # different hat. They are flagged for exclusion instead.
    assert frame["in_warmup"].any()
    assert frame.loc[frame["in_warmup"], "carrier_hist_mean_transit"].isna().any()


def test_calendar_features_are_derived_correctly():
    frame = build_features_pandas(make_shipments(rows=40))
    row = frame.iloc[0]

    dispatch = pd.to_datetime(row["dispatch_ts"])
    assert row["dispatch_dow"] == dispatch.dayofweek
    assert row["dispatch_month"] == dispatch.month
    assert row["is_weekend_dispatch"] == int(dispatch.dayofweek >= 5)
    assert row["service_level_factor"] == 1.0


@pytest.mark.skipif(not spark_available(), reason="No JVM or pyspark available")
def test_spark_and_pandas_engines_agree():
    from pipeline.features import build_features_spark

    shipments = make_shipments(rows=80)
    from_pandas = build_features_pandas(shipments)
    from_spark = build_features_spark(shipments)

    merged = from_pandas.merge(from_spark, on="shipment_id", suffixes=("_p", "_s"))
    assert len(merged) == len(shipments)

    for column in ("carrier_hist_mean_transit", "carrier_hist_late_rate",
                   "warehouse_hist_mean_transit", "lane_hist_shipments",
                   "dispatch_dow", "dispatch_month", "service_level_factor"):
        left = merged[f"{column}_p"].to_numpy(dtype=float)
        right = merged[f"{column}_s"].to_numpy(dtype=float)
        assert np.allclose(left, right, equal_nan=True, rtol=1e-6), column
