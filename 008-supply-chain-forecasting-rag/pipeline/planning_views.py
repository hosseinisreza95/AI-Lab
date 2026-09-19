"""Inventory forecasting and the planning views the regional teams consume.

Two outputs:

- `store_cover.parquet` - forecast weekly demand per store and category, joined
  against stock on hand and stock in transit, expressed as days of cover.
- `lane_performance.parquet` - on-time rate, median transit and volume per lane
  and carrier, which is what a regional planner looks at before renegotiating.

Demand is forecast with a seasonal decomposition rather than a learned model:
weekday shape times a recent level, with a peak-season multiplier. On 20 stores x
5 categories that is 100 short series, most of which do not have enough signal to
justify anything heavier, and a per-series SARIMA would take longer to run than
the whole rest of the pipeline.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CURATED_DIR = BASE_DIR / "data" / "curated"

RECENT_WINDOW_DAYS = 56
PEAK_MONTHS = {11, 12}
PEAK_UPLIFT = 1.45

# A lane needs this many shipments in the window before its on-time rate is
# treated as a measurement rather than an anecdote. Without it the ranking is
# topped by one-off cross-dock lanes that ran once and ran late.
MIN_LANE_SHIPMENTS = 20


def forecast_store_demand(horizon_days: int = 21) -> pd.DataFrame:
    """Per store and category: expected daily units over the horizon."""
    demand = pd.read_parquet(RAW_DIR / "store_demand.parquet")
    demand["sales_date"] = pd.to_datetime(demand["sales_date"])

    last_date = demand["sales_date"].max()
    recent = demand[demand["sales_date"] > last_date - pd.Timedelta(days=RECENT_WINDOW_DAYS)].copy()
    recent["dow"] = recent["sales_date"].dt.dayofweek

    level = recent.groupby(["store_id", "category"])["units_sold"].mean().rename("level")

    # Weekday shape is estimated across all stores. A single store-category has
    # only eight observations per weekday in the window, which is not enough to
    # estimate a shape from; the shape is a property of shopping behaviour, not
    # of the individual store.
    overall = recent["units_sold"].mean()
    dow_shape = (recent.groupby("dow")["units_sold"].mean() / overall).rename("dow_factor")

    future = pd.DataFrame({
        "forecast_date": [last_date + pd.Timedelta(days=offset) for offset in range(1, horizon_days + 1)]
    })
    future["dow"] = future["forecast_date"].dt.dayofweek
    future["month"] = future["forecast_date"].dt.month
    future = future.merge(dow_shape, on="dow", how="left")
    future["peak_factor"] = np.where(future["month"].isin(PEAK_MONTHS), PEAK_UPLIFT, 1.0)

    keys = level.reset_index()
    forecast = keys.merge(future, how="cross")
    forecast["forecast_units"] = (
        forecast["level"] * forecast["dow_factor"] * forecast["peak_factor"]
    ).round(2)

    return forecast[
        ["store_id", "category", "forecast_date", "forecast_units", "level"]
    ]


def build_store_cover(horizon_days: int = 21) -> pd.DataFrame:
    """Days of cover per store and category, and the date it runs out."""
    forecast = forecast_store_demand(horizon_days)
    inventory = pd.read_parquet(RAW_DIR / "store_inventory.parquet")
    stores = pd.read_parquet(RAW_DIR / "stores.parquet")

    daily = forecast.groupby(["store_id", "category"]).agg(
        forecast_daily_units=("forecast_units", "mean"),
        horizon_units=("forecast_units", "sum"),
    ).reset_index()

    cover = (
        inventory.merge(daily, on=["store_id", "category"], how="left")
        .merge(stores[["store_id", "store_name", "region", "default_warehouse"]], on="store_id", how="left")
    )
    cover["available_units"] = cover["units_on_hand"] + cover["units_in_transit"]
    cover["days_of_cover"] = np.where(
        cover["forecast_daily_units"] > 0,
        (cover["available_units"] / cover["forecast_daily_units"]).round(1),
        np.inf,
    )
    cover["shortfall_units"] = (cover["horizon_units"] - cover["available_units"]).round(0).clip(lower=0)

    today = pd.Timestamp(dt.date.today())
    cover["projected_stockout_date"] = np.where(
        np.isfinite(cover["days_of_cover"]) & (cover["days_of_cover"] <= horizon_days),
        (today + pd.to_timedelta(cover["days_of_cover"].replace(np.inf, 0), unit="D")).dt.date.astype(str),
        None,
    )
    cover["risk"] = pd.cut(
        cover["days_of_cover"].replace(np.inf, 999),
        bins=[-0.01, 3, 7, 14, np.inf],
        labels=["critical", "high", "medium", "ok"],
    ).astype(str)

    return cover.sort_values("days_of_cover")[
        ["store_id", "store_name", "region", "default_warehouse", "category",
         "units_on_hand", "units_in_transit", "forecast_daily_units",
         "days_of_cover", "projected_stockout_date", "shortfall_units", "risk"]
    ].reset_index(drop=True)


def build_lane_performance() -> pd.DataFrame:
    """On-time rate and transit statistics per lane and carrier."""
    shipments = pd.read_parquet(RAW_DIR / "shipments.parquet")
    shipments["was_late"] = (
        pd.to_datetime(shipments["actual_arrival_ts"]) > pd.to_datetime(shipments["promised_ts"])
    )
    shipments["lane"] = shipments["origin_warehouse"] + "->" + shipments["destination_store"]

    # Last 90 days only. A lane's on-time rate over 18 months hides exactly the
    # degradation a planner is looking for.
    cutoff = pd.to_datetime(shipments["dispatch_ts"]).max() - pd.Timedelta(days=90)
    recent = shipments[pd.to_datetime(shipments["dispatch_ts"]) >= cutoff]

    performance = recent.groupby(["lane", "carrier"]).agg(
        shipments=("shipment_id", "count"),
        median_transit_days=("actual_transit_days", "median"),
        p90_transit_days=("actual_transit_days", lambda values: float(np.percentile(values, 90))),
        on_time_rate_pct=("was_late", lambda values: round(float(1 - values.mean()) * 100, 1)),
        avg_units=("units", "mean"),
    ).reset_index()

    performance["median_transit_days"] = performance["median_transit_days"].round(2)
    performance["p90_transit_days"] = performance["p90_transit_days"].round(2)
    performance["avg_units"] = performance["avg_units"].round(0)
    performance["sufficient_volume"] = performance["shipments"] >= MIN_LANE_SHIPMENTS

    # Rank the measurable lanes first, worst on-time rate at the top. Low-volume
    # lanes stay in the table so nothing disappears, but they never lead it.
    return performance.sort_values(
        ["sufficient_volume", "on_time_rate_pct", "shipments"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def build_all(horizon_days: int = 21) -> dict:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    cover = build_store_cover(horizon_days)
    performance = build_lane_performance()

    cover.to_parquet(CURATED_DIR / "store_cover.parquet", index=False)
    performance.to_parquet(CURATED_DIR / "lane_performance.parquet", index=False)

    measurable = performance[performance["sufficient_volume"]]
    return {
        "store_cover_rows": len(cover),
        "at_risk_store_categories": int((cover["risk"].isin(["critical", "high"])).sum()),
        "lane_performance_rows": len(performance),
        "lanes_with_sufficient_volume": int(len(measurable)),
        "worst_measurable_lane_on_time_pct": float(measurable["on_time_rate_pct"].min()),
        "median_measurable_lane_on_time_pct": round(
            float(measurable["on_time_rate_pct"].median()), 1
        ),
    }


if __name__ == "__main__":
    import json

    print("Building planning views ...")
    print(json.dumps(build_all(), indent=2))

    cover = pd.read_parquet(CURATED_DIR / "store_cover.parquet")
    print("\nStores closest to running out:")
    print(cover.head(8)[
        ["store_name", "category", "units_on_hand", "forecast_daily_units",
         "days_of_cover", "risk"]
    ].to_string(index=False))

    performance = pd.read_parquet(CURATED_DIR / "lane_performance.parquet")
    print("\nWorst-performing lanes (last 90 days, min "
          f"{MIN_LANE_SHIPMENTS} shipments):")
    measurable = performance[performance["sufficient_volume"]]
    print(measurable.head(8).drop(columns=["sufficient_volume"]).to_string(index=False))
