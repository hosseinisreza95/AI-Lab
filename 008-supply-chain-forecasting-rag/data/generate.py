"""Generate a synthetic but structured shipment dataset.

The delivery-time model is only worth building if transit time is actually a
function of something. The generator therefore encodes real drivers:

- lane distance, which dominates
- carrier quality, which differs systematically and drifts over time
- warehouse congestion, which is worse on Mondays and in peak season
- service level, which buys priority handling
- weather and customs shocks in specific regions and months

and then adds noise on top. A model that cannot recover those relationships is
not being defeated by a hard problem, it is broken.

Run with:  python -m data.generate
"""
from __future__ import annotations

import datetime as dt
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
SEED = 20260919

HISTORY_DAYS = 540          # ~18 months of completed deliveries
LIVE_SHIPMENTS = 240        # currently in transit, for the tracking tool

WAREHOUSES = {
    "WH-LYO": {"name": "Lyon DC", "region": "FR-South", "country": "FR", "capacity": 1400},
    "WH-LIL": {"name": "Lille DC", "region": "FR-North", "country": "FR", "capacity": 1100},
    "WH-BCN": {"name": "Barcelona DC", "region": "ES", "country": "ES", "capacity": 900},
    "WH-MIL": {"name": "Milan DC", "region": "IT", "country": "IT", "capacity": 950},
    "WH-KRK": {"name": "Krakow DC", "region": "PL", "country": "PL", "capacity": 800},
}

# Stores, with the warehouse that normally serves them and the road distance.
STORES = [
    ("ST-0101", "Lyon Part-Dieu", "FR-South", "WH-LYO", 12),
    ("ST-0102", "Grenoble", "FR-South", "WH-LYO", 110),
    ("ST-0103", "Marseille", "FR-South", "WH-LYO", 315),
    ("ST-0104", "Toulouse", "FR-South", "WH-LYO", 540),
    ("ST-0105", "Nice", "FR-South", "WH-LYO", 470),
    ("ST-0201", "Lille Centre", "FR-North", "WH-LIL", 8),
    ("ST-0202", "Paris La Defense", "FR-North", "WH-LIL", 225),
    ("ST-0203", "Rouen", "FR-North", "WH-LIL", 250),
    ("ST-0204", "Strasbourg", "FR-North", "WH-LIL", 520),
    ("ST-0301", "Barcelona Diagonal", "ES", "WH-BCN", 10),
    ("ST-0302", "Valencia", "ES", "WH-BCN", 350),
    ("ST-0303", "Madrid", "ES", "WH-BCN", 620),
    ("ST-0304", "Zaragoza", "ES", "WH-BCN", 300),
    ("ST-0401", "Milan Duomo", "IT", "WH-MIL", 9),
    ("ST-0402", "Turin", "IT", "WH-MIL", 145),
    ("ST-0403", "Bologna", "IT", "WH-MIL", 215),
    ("ST-0404", "Rome", "IT", "WH-MIL", 575),
    ("ST-0501", "Krakow Galeria", "PL", "WH-KRK", 7),
    ("ST-0502", "Warsaw", "PL", "WH-KRK", 295),
    ("ST-0503", "Wroclaw", "PL", "WH-KRK", 270),
]

# base_speed_kmpd: how far this carrier covers in a transit day.
# reliability: lower is better, it scales the noise.
CARRIERS = {
    "CAR-ALP": {"name": "Alpine Freight", "base_speed_kmpd": 620, "reliability": 0.10, "share": 0.30},
    "CAR-MER": {"name": "Meridian Logistics", "base_speed_kmpd": 540, "reliability": 0.16, "share": 0.26},
    "CAR-EUR": {"name": "EuroLink Transport", "base_speed_kmpd": 480, "reliability": 0.24, "share": 0.22},
    "CAR-NOR": {"name": "Nordstar Cargo", "base_speed_kmpd": 700, "reliability": 0.08, "share": 0.12},
    "CAR-VIA": {"name": "ViaTerra", "base_speed_kmpd": 430, "reliability": 0.30, "share": 0.10},
}

SERVICE_LEVELS = {"express": 0.72, "standard": 1.0, "economy": 1.35}
CATEGORIES = ["apparel", "footwear", "equipment", "accessories", "nutrition"]

# Disruptions the model should be able to pick up through its date features.
PEAK_MONTHS = {11, 12}
WEATHER_MONTHS = {1, 2}


def _carrier_drift(carrier: str, day_index: int) -> float:
    """Carrier performance is not static. EuroLink degrades over the period and
    ViaTerra improves, which is the kind of drift that makes a stale
    carrier-reliability feature dangerous."""
    progress = day_index / HISTORY_DAYS
    if carrier == "CAR-EUR":
        return 1.0 + 0.22 * progress
    if carrier == "CAR-VIA":
        return 1.0 - 0.18 * progress
    return 1.0


def _transit_days(
    distance_km: int,
    carrier: str,
    service_level: str,
    dispatch: dt.date,
    day_index: int,
    warehouse_load: float,
    rng: random.Random,
) -> float:
    spec = CARRIERS[carrier]
    base = distance_km / spec["base_speed_kmpd"]
    base *= SERVICE_LEVELS[service_level]
    base *= _carrier_drift(carrier, day_index)

    # Warehouse congestion: dispatch takes longer when the DC is running hot.
    base += 0.55 * max(0.0, warehouse_load - 0.80)

    if dispatch.month in PEAK_MONTHS:
        base *= 1.18
    if dispatch.month in WEATHER_MONTHS:
        base *= 1.09

    # Weekend effect: nothing moves much on Sunday.
    if dispatch.weekday() == 5:
        base += 0.7
    elif dispatch.weekday() == 6:
        base += 1.2

    base *= max(0.35, rng.gauss(1.0, spec["reliability"]))

    # Rare severe disruption: a border closure, a strike, a failed handover.
    if rng.random() < 0.012:
        base += rng.uniform(1.5, 4.0)

    return max(0.4, base)


def _demand(store_index: int, category: str, day: dt.date, rng: random.Random) -> int:
    base = 18 + (store_index % 7) * 4
    base *= {"apparel": 1.4, "footwear": 1.0, "equipment": 0.7,
             "accessories": 1.2, "nutrition": 0.6}[category]
    yearly = 1 + 0.22 * math.sin(2 * math.pi * (day.timetuple().tm_yday - 60) / 365)
    weekday = {0: 0.85, 1: 0.9, 2: 0.95, 3: 1.0, 4: 1.25, 5: 1.5, 6: 0.8}[day.weekday()]
    peak = 1.45 if day.month in PEAK_MONTHS else 1.0
    return max(0, int(round(base * yearly * weekday * peak * rng.gauss(1.0, 0.18))))


def generate(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    today = dt.date.today()
    start = today - dt.timedelta(days=HISTORY_DAYS)

    carrier_ids = list(CARRIERS)
    carrier_weights = [CARRIERS[c]["share"] for c in carrier_ids]
    service_ids = list(SERVICE_LEVELS)

    # --- daily warehouse load, needed before shipments so it can drive them ---
    load_by_day: dict[tuple[str, dt.date], float] = {}
    for offset in range(HISTORY_DAYS + 30):
        day = start + dt.timedelta(days=offset)
        for warehouse in WAREHOUSES:
            seasonal = 1.3 if day.month in PEAK_MONTHS else 1.0
            weekday = 1.15 if day.weekday() == 0 else (0.5 if day.weekday() == 6 else 1.0)
            load_by_day[(warehouse, day)] = min(
                1.35, 0.68 * seasonal * weekday * rng.gauss(1.0, 0.09)
            )

    # --- completed shipments -------------------------------------------------
    shipments = []
    shipment_counter = 0
    for offset in range(HISTORY_DAYS):
        dispatch = start + dt.timedelta(days=offset)
        volume = int(rng.gauss(95, 18) * (1.35 if dispatch.month in PEAK_MONTHS else 1.0))
        volume = max(20, volume)

        for _ in range(volume):
            store_index = rng.randrange(len(STORES))
            store_id, _store_name, region, warehouse, distance = STORES[store_index]

            # 12% of shipments are cross-docked from a different DC, which makes
            # the lane feature genuinely informative rather than a proxy for store.
            if rng.random() < 0.12:
                warehouse = rng.choice([w for w in WAREHOUSES if w != warehouse])
                distance = int(distance * rng.uniform(1.4, 2.6))

            carrier = rng.choices(carrier_ids, weights=carrier_weights)[0]
            service_level = rng.choices(service_ids, weights=[0.18, 0.62, 0.20])[0]
            load = load_by_day[(warehouse, dispatch)]

            actual = _transit_days(
                distance, carrier, service_level, dispatch, offset, load, rng
            )
            # What the planning system promised, before this model existed: a
            # crude distance rule plus a service-level adjustment.
            planned = max(1, math.ceil(distance / 550 * SERVICE_LEVELS[service_level]))

            shipment_counter += 1
            dispatch_ts = dt.datetime.combine(dispatch, dt.time(rng.randrange(6, 20), rng.randrange(60)))
            arrival_ts = dispatch_ts + dt.timedelta(days=actual)

            shipments.append({
                "shipment_id": f"SHP-{shipment_counter:07d}",
                "order_id": f"ORD-{rng.randrange(100000, 999999)}",
                "origin_warehouse": warehouse,
                "destination_store": store_id,
                "region": region,
                "carrier": carrier,
                "service_level": service_level,
                "category": rng.choice(CATEGORIES),
                "distance_km": distance,
                "units": max(1, int(rng.gauss(310, 120))),
                "weight_kg": round(max(5.0, rng.gauss(820, 300)), 1),
                "warehouse_load": round(load, 3),
                "dispatch_ts": dispatch_ts,
                "promised_ts": dispatch_ts + dt.timedelta(days=planned),
                "actual_arrival_ts": arrival_ts,
                "planned_transit_days": planned,
                "actual_transit_days": round(actual, 3),
                "status": "delivered",
            })

    # --- live shipments, for the tracking tool -------------------------------
    live = []
    checkpoints = ["departed origin DC", "in transit", "at line-haul hub",
                   "out for delivery", "held at customs", "delayed - weather"]
    for index in range(LIVE_SHIPMENTS):
        store_index = rng.randrange(len(STORES))
        store_id, _store_name, region, warehouse, distance = STORES[store_index]
        carrier = rng.choices(carrier_ids, weights=carrier_weights)[0]
        service_level = rng.choices(service_ids, weights=[0.18, 0.62, 0.20])[0]

        days_out = rng.uniform(0.1, 3.5)
        dispatch_ts = dt.datetime.now() - dt.timedelta(days=days_out)
        planned = max(1, math.ceil(distance / 550 * SERVICE_LEVELS[service_level]))
        load = load_by_day[(warehouse, dispatch_ts.date())]
        expected = _transit_days(distance, carrier, service_level,
                                 dispatch_ts.date(), HISTORY_DAYS, load, rng)

        progress = min(0.97, days_out / max(expected, 0.5))
        checkpoint = (
            checkpoints[0] if progress < 0.15
            else checkpoints[3] if progress > 0.85
            else rng.choice(checkpoints[1:3] + (checkpoints[4:] if rng.random() < 0.18 else []))
        )

        shipment_counter += 1
        live.append({
            "shipment_id": f"SHP-{shipment_counter:07d}",
            "order_id": f"ORD-{rng.randrange(100000, 999999)}",
            "origin_warehouse": warehouse,
            "destination_store": store_id,
            "region": region,
            "carrier": carrier,
            "service_level": service_level,
            "category": rng.choice(CATEGORIES),
            "distance_km": distance,
            "units": max(1, int(rng.gauss(310, 120))),
            "weight_kg": round(max(5.0, rng.gauss(820, 300)), 1),
            "warehouse_load": round(load, 3),
            "dispatch_ts": dispatch_ts,
            "promised_ts": dispatch_ts + dt.timedelta(days=planned),
            "actual_arrival_ts": pd.NaT,
            "planned_transit_days": planned,
            "actual_transit_days": np.nan,
            "status": "delayed" if "delayed" in checkpoint or "customs" in checkpoint else "in_transit",
            "last_checkpoint": checkpoint,
            "last_scan_ts": dt.datetime.now() - dt.timedelta(hours=rng.uniform(0.5, 14)),
            "progress_pct": round(progress * 100, 1),
        })

    # --- store demand, for the inventory forecast ----------------------------
    demand_rows = []
    for offset in range(HISTORY_DAYS):
        day = start + dt.timedelta(days=offset)
        for store_index, (store_id, _name, region, _wh, _dist) in enumerate(STORES):
            for category in CATEGORIES:
                demand_rows.append({
                    "store_id": store_id,
                    "region": region,
                    "category": category,
                    "sales_date": day,
                    "units_sold": _demand(store_index, category, day, rng),
                })

    # --- current store inventory --------------------------------------------
    demand_frame = pd.DataFrame(demand_rows)
    recent = demand_frame[demand_frame["sales_date"] >= today - dt.timedelta(days=28)]
    recent_rate = recent.groupby(["store_id", "category"])["units_sold"].mean()

    inventory_rows = []
    for (store_id, category), rate in recent_rate.items():
        inventory_rows.append({
            "store_id": store_id,
            "category": category,
            "units_on_hand": max(0, int(rate * rng.uniform(4.0, 26.0))),
            "units_in_transit": max(0, int(rate * rng.uniform(0.0, 6.0))),
            "counted_at": dt.datetime.now(),
        })

    frames = {
        "shipments": pd.DataFrame(shipments),
        "live_shipments": pd.DataFrame(live),
        "store_demand": demand_frame,
        "store_inventory": pd.DataFrame(inventory_rows),
        "stores": pd.DataFrame(
            [
                {"store_id": s, "store_name": n, "region": r,
                 "default_warehouse": w, "distance_km": d}
                for s, n, r, w, d in STORES
            ]
        ),
        "warehouses": pd.DataFrame(
            [{"warehouse_id": k, **v} for k, v in WAREHOUSES.items()]
        ),
        "carriers": pd.DataFrame(
            [{"carrier_id": k, "carrier_name": v["name"]} for k, v in CARRIERS.items()]
        ),
    }

    for name, frame in frames.items():
        # Parquet is what the Spark stage reads; it keeps dtypes and is what the
        # real pipeline uses on object storage.
        frame.to_parquet(RAW_DIR / f"{name}.parquet", index=False)

    return {name: len(frame) for name, frame in frames.items()}


if __name__ == "__main__":
    import json

    print("Generating supply chain dataset ...")
    print(json.dumps(generate(), indent=2))
    print(f"\nWritten to {RAW_DIR}")
