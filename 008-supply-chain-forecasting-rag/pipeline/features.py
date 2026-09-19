"""Feature engineering, in two interchangeable engines.

The production pipeline runs on Spark because the real shipment table is large
enough to need it. Spark needs a JVM, which makes the experiment unrunnable for
anyone who just wants to read the code, so the same features are implemented in
pandas and the engine is selected at runtime. `test_features.py` asserts the two
produce identical output on the same input, which is the only thing that makes a
dual implementation safe.

The feature that matters most here is carrier and lane historical performance,
and it is the one that is easiest to get wrong. Computing a carrier's average
transit time over the whole dataset and joining it onto every row leaks the
future into the past: the training rows then carry information from shipments
that had not happened yet, the offline metric looks excellent, and the model
degrades the moment it sees real traffic. Every historical aggregate below is
computed over a strictly prior window.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CURATED_DIR = BASE_DIR / "data" / "curated"

# Rows in the first `WARMUP_DAYS` have no usable history and are dropped from
# training rather than filled with a global mean, which would reintroduce the
# leak by the back door.
WARMUP_DAYS = 60
PEAK_MONTHS = {11, 12}
WEATHER_MONTHS = {1, 2}

FEATURE_COLUMNS = [
    "distance_km",
    "units",
    "weight_kg",
    "warehouse_load",
    "planned_transit_days",
    "dispatch_dow",
    "dispatch_month",
    "is_peak_season",
    "is_weather_season",
    "is_weekend_dispatch",
    "carrier_hist_mean_transit",
    "carrier_hist_late_rate",
    "lane_hist_median_transit",
    "lane_hist_shipments",
    "warehouse_hist_mean_transit",
    "service_level_factor",
]
CATEGORICAL_COLUMNS = ["carrier", "service_level", "origin_warehouse", "region"]
TARGET = "actual_transit_days"

SERVICE_LEVEL_FACTOR = {"express": 0.72, "standard": 1.0, "economy": 1.35}


def spark_available() -> bool:
    """Spark is used when it is importable and a JVM is present."""
    if os.getenv("FORCE_PANDAS", "").lower() in {"1", "true", "yes"}:
        return False
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return bool(os.getenv("JAVA_HOME") or os.getenv("SPARK_HOME"))


# ---------------------------------------------------------------------------
# pandas engine
# ---------------------------------------------------------------------------

def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    dispatch = pd.to_datetime(frame["dispatch_ts"])
    frame["dispatch_date"] = dispatch.dt.normalize()
    frame["dispatch_dow"] = dispatch.dt.dayofweek
    frame["dispatch_month"] = dispatch.dt.month
    frame["is_peak_season"] = frame["dispatch_month"].isin(PEAK_MONTHS).astype(int)
    frame["is_weather_season"] = frame["dispatch_month"].isin(WEATHER_MONTHS).astype(int)
    frame["is_weekend_dispatch"] = (frame["dispatch_dow"] >= 5).astype(int)
    frame["service_level_factor"] = frame["service_level"].map(SERVICE_LEVEL_FACTOR).fillna(1.0)
    return frame


def _expanding_prior_mean(frame: pd.DataFrame, keys: list[str], value: str) -> pd.Series:
    """Mean of `value` over all EARLIER rows within each key group.

    Sorting by dispatch date and taking a shifted expanding mean is what keeps a
    row's own outcome, and every later outcome, out of its own feature.
    """
    ordered = frame.sort_values(["dispatch_date", "shipment_id"])
    grouped = ordered.groupby(keys, observed=True)[value]
    prior = grouped.transform(lambda series: series.shift(1).expanding().mean())
    return prior.reindex(frame.index)


def _expanding_prior_median(frame: pd.DataFrame, keys: list[str], value: str) -> pd.Series:
    ordered = frame.sort_values(["dispatch_date", "shipment_id"])
    grouped = ordered.groupby(keys, observed=True)[value]
    prior = grouped.transform(lambda series: series.shift(1).expanding().median())
    return prior.reindex(frame.index)


def _expanding_prior_count(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    ordered = frame.sort_values(["dispatch_date", "shipment_id"])
    grouped = ordered.groupby(keys, observed=True)["shipment_id"]
    prior = grouped.transform(lambda series: series.shift(1).expanding().count())
    return prior.reindex(frame.index)


def build_features_pandas(shipments: pd.DataFrame) -> pd.DataFrame:
    frame = shipments.copy()
    frame = _calendar_features(frame)
    frame["lane"] = frame["origin_warehouse"] + "->" + frame["destination_store"]
    frame["was_late"] = (
        pd.to_datetime(frame["actual_arrival_ts"]) > pd.to_datetime(frame["promised_ts"])
    ).astype(float)

    frame["carrier_hist_mean_transit"] = _expanding_prior_mean(
        frame, ["carrier"], TARGET
    )
    frame["carrier_hist_late_rate"] = _expanding_prior_mean(
        frame, ["carrier"], "was_late"
    )
    frame["lane_hist_median_transit"] = _expanding_prior_median(
        frame, ["lane"], TARGET
    )
    frame["lane_hist_shipments"] = _expanding_prior_count(frame, ["lane"]).fillna(0)
    frame["warehouse_hist_mean_transit"] = _expanding_prior_mean(
        frame, ["origin_warehouse"], TARGET
    )

    cutoff = frame["dispatch_date"].min() + pd.Timedelta(days=WARMUP_DAYS)
    frame["in_warmup"] = frame["dispatch_date"] < cutoff

    return frame.sort_values(["dispatch_date", "shipment_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spark engine
# ---------------------------------------------------------------------------

def build_features_spark(shipments: pd.DataFrame) -> pd.DataFrame:
    """Same features, computed with Spark window functions.

    Returns pandas so the caller cannot tell which engine ran. On a real cluster
    this would write Parquet and never collect, but the contract - identical
    columns, identical values - is what the parity test pins down.
    """
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.appName("supply-chain-features")
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
        .getOrCreate()
    )

    sdf = spark.createDataFrame(shipments)
    sdf = (
        sdf.withColumn("dispatch_date", F.to_date("dispatch_ts"))
        .withColumn("dispatch_dow", F.dayofweek("dispatch_ts") - 1)
        .withColumn("dispatch_month", F.month("dispatch_ts"))
        .withColumn("is_peak_season", F.col("dispatch_month").isin(list(PEAK_MONTHS)).cast("int"))
        .withColumn("is_weather_season", F.col("dispatch_month").isin(list(WEATHER_MONTHS)).cast("int"))
        .withColumn("is_weekend_dispatch", (F.col("dispatch_dow") >= 5).cast("int"))
        .withColumn("lane", F.concat_ws("->", "origin_warehouse", "destination_store"))
        .withColumn("was_late", (F.col("actual_arrival_ts") > F.col("promised_ts")).cast("double"))
    )

    factor = F.create_map(
        *[item for pair in SERVICE_LEVEL_FACTOR.items() for item in (F.lit(pair[0]), F.lit(pair[1]))]
    )
    sdf = sdf.withColumn("service_level_factor", F.coalesce(factor[F.col("service_level")], F.lit(1.0)))

    def prior(partition: str):
        # rowsBetween(unboundedPreceding, -1) is the Spark spelling of "every
        # earlier row and not this one", which is the whole leakage guard.
        return (
            Window.partitionBy(partition)
            .orderBy("dispatch_date", "shipment_id")
            .rowsBetween(Window.unboundedPreceding, -1)
        )

    sdf = (
        sdf.withColumn("carrier_hist_mean_transit", F.avg(TARGET).over(prior("carrier")))
        .withColumn("carrier_hist_late_rate", F.avg("was_late").over(prior("carrier")))
        .withColumn(
            "lane_hist_median_transit",
            F.expr(f"percentile_approx({TARGET}, 0.5, 10000)").over(prior("lane")),
        )
        .withColumn("lane_hist_shipments", F.count("shipment_id").over(prior("lane")))
        .withColumn(
            "warehouse_hist_mean_transit", F.avg(TARGET).over(prior("origin_warehouse"))
        )
    )

    frame = sdf.toPandas()
    spark.stop()

    frame["dispatch_date"] = pd.to_datetime(frame["dispatch_date"])
    cutoff = frame["dispatch_date"].min() + pd.Timedelta(days=WARMUP_DAYS)
    frame["in_warmup"] = frame["dispatch_date"] < cutoff
    frame["lane_hist_shipments"] = frame["lane_hist_shipments"].fillna(0)

    return frame.sort_values(["dispatch_date", "shipment_id"]).reset_index(drop=True)


def build_features(shipments: pd.DataFrame | None = None, engine: str | None = None) -> pd.DataFrame:
    if shipments is None:
        shipments = pd.read_parquet(RAW_DIR / "shipments.parquet")

    engine = engine or ("spark" if spark_available() else "pandas")
    frame = build_features_spark(shipments) if engine == "spark" else build_features_pandas(shipments)
    frame.attrs["engine"] = engine
    return frame


def write_curated(frame: pd.DataFrame) -> Path:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    path = CURATED_DIR / "shipment_features.parquet"
    keep = [
        "shipment_id", "order_id", "dispatch_date", "destination_store", "lane",
        "carrier", "service_level", "origin_warehouse", "region", "in_warmup",
        "promised_ts", "actual_arrival_ts", TARGET, *FEATURE_COLUMNS,
    ]
    frame[[column for column in keep if column in frame.columns]].to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    engine = "spark" if spark_available() else "pandas"
    print(f"Building shipment features with the {engine} engine ...")
    features = build_features(engine=engine)
    path = write_curated(features)
    usable = int((~features["in_warmup"]).sum())
    print(
        f"  {len(features):,} rows, {usable:,} usable after the {WARMUP_DAYS}-day warm-up\n"
        f"  written to {path}"
    )
    print(
        features[["carrier", "carrier_hist_mean_transit", "lane_hist_median_transit",
                  "lane_hist_shipments"]].tail(5).to_string(index=False)
    )
