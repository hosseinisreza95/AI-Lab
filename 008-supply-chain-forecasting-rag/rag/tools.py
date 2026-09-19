"""Tools for the logistics assistant.

"Where is this truck?" is not a document question, and answering it with document
retrieval alone is the mistake this design avoids. The live answer lives in the
shipment table; the procedure for what to do about it lives in the SOPs. The
agent gets both kinds of tool and decides which the question needs - and for most
real questions the answer needs both, because "it is 26 hours late" is only
useful next to "26 hours late is Tier 2, contact the carrier account manager".
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CURATED_DIR = BASE_DIR / "data" / "curated"
KB_DIR = BASE_DIR / "knowledge_base"
MODEL_DIR = BASE_DIR / "models"

_cache: dict[str, object] = {}


def _load(name: str, directory: Path = RAW_DIR) -> pd.DataFrame:
    key = f"{directory.name}/{name}"
    if key not in _cache:
        path = directory / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run the pipeline first: "
                f"python -m data.generate && python -m pipeline.features && "
                f"python -m pipeline.planning_views"
            )
        _cache[key] = pd.read_parquet(path)
    return _cache[key].copy()


def _predict_arrival(row: pd.Series) -> dict:
    """Predicted arrival for a live shipment, from the trained model when it
    exists and from lane history otherwise."""
    dispatch = pd.to_datetime(row["dispatch_ts"])
    promised = pd.to_datetime(row["promised_ts"])

    model_path = MODEL_DIR / "delivery_time_model.joblib"
    predicted_days = None

    if model_path.exists():
        try:
            import joblib
            from sklearn.preprocessing import OrdinalEncoder  # noqa: F401

            bundle = joblib.load(model_path)
            features = _load("shipment_features", CURATED_DIR)
            lane = f"{row['origin_warehouse']}->{row['destination_store']}"
            lane_history = features[features["lane"] == lane]
            carrier_history = features[features["carrier"] == row["carrier"]]

            values = {
                "distance_km": row["distance_km"],
                "units": row["units"],
                "weight_kg": row["weight_kg"],
                "warehouse_load": row["warehouse_load"],
                "planned_transit_days": row["planned_transit_days"],
                "dispatch_dow": dispatch.dayofweek,
                "dispatch_month": dispatch.month,
                "is_peak_season": int(dispatch.month in {11, 12}),
                "is_weather_season": int(dispatch.month in {1, 2}),
                "is_weekend_dispatch": int(dispatch.dayofweek >= 5),
                "carrier_hist_mean_transit": float(carrier_history["actual_transit_days"].mean()),
                "carrier_hist_late_rate": float(
                    (pd.to_datetime(carrier_history["actual_arrival_ts"])
                     > pd.to_datetime(carrier_history["promised_ts"])).mean()
                ),
                "lane_hist_median_transit": float(lane_history["actual_transit_days"].median())
                if len(lane_history) else float(features["actual_transit_days"].median()),
                "lane_hist_shipments": float(len(lane_history)),
                "warehouse_hist_mean_transit": float(
                    features[features["origin_warehouse"] == row["origin_warehouse"]][
                        "actual_transit_days"
                    ].mean()
                ),
                "service_level_factor": {"express": 0.72, "standard": 1.0, "economy": 1.35}.get(
                    row["service_level"], 1.0
                ),
            }
            numeric = np.array([[values[name] for name in bundle["feature_columns"]]], dtype=float)
            # Categoricals were ordinal-encoded at train time. Without the fitted
            # encoder here, -1 is the documented "unseen category" value, which is
            # what the model was trained to tolerate.
            categorical = np.full((1, len(bundle["categorical_columns"])), -1.0)
            predicted_days = float(
                bundle["model"].predict(np.hstack([numeric, categorical]))[0]
            )
        except Exception:
            predicted_days = None

    if predicted_days is None:
        shipments = _load("shipments")
        lane_mask = (
            (shipments["origin_warehouse"] == row["origin_warehouse"])
            & (shipments["destination_store"] == row["destination_store"])
        )
        lane = shipments[lane_mask]
        predicted_days = float(
            lane["actual_transit_days"].median()
            if len(lane) >= 5
            else shipments["actual_transit_days"].median()
        )
        source = "lane history median (model not trained)"
    else:
        source = "delivery-time model"

    predicted_arrival = dispatch + pd.Timedelta(days=predicted_days)
    delay_hours = (predicted_arrival - promised).total_seconds() / 3600

    return {
        "predicted_arrival": predicted_arrival.strftime("%Y-%m-%d %H:%M"),
        "predicted_transit_days": round(predicted_days, 2),
        "promised_arrival": promised.strftime("%Y-%m-%d %H:%M"),
        "projected_delay_hours": round(delay_hours, 1),
        "projected_to_miss_promise": bool(delay_hours > 0),
        "prediction_source": source,
    }


@tool
def track_shipment(identifier: str) -> str:
    """Look up a live shipment by shipment ID (SHP-...) or order ID (ORD-...).

    Use for "where is this truck", "where is this delivery", "when will order X
    arrive". Returns current checkpoint, last scan, progress, the promised date
    and the model's predicted arrival.
    """
    needle = identifier.strip().upper()
    live = _load("live_shipments")
    match = live[(live["shipment_id"] == needle) | (live["order_id"] == needle)]

    if match.empty:
        delivered = _load("shipments")
        done = delivered[
            (delivered["shipment_id"] == needle) | (delivered["order_id"] == needle)
        ]
        if not done.empty:
            row = done.iloc[0]
            arrival = pd.to_datetime(row["actual_arrival_ts"])
            promised = pd.to_datetime(row["promised_ts"])
            return json.dumps({
                "shipment_id": row["shipment_id"],
                "status": "delivered",
                "destination_store": row["destination_store"],
                "carrier": row["carrier"],
                "delivered_at": arrival.strftime("%Y-%m-%d %H:%M"),
                "promised_at": promised.strftime("%Y-%m-%d %H:%M"),
                "was_late": bool(arrival > promised),
                "delay_hours": round((arrival - promised).total_seconds() / 3600, 1),
            })
        return json.dumps({
            "error": f"No shipment or order matching '{identifier}'.",
            "hint": "Identifiers look like SHP-0053276 or ORD-481923.",
        })

    row = match.iloc[0]
    return json.dumps({
        "shipment_id": row["shipment_id"],
        "order_id": row["order_id"],
        "status": row["status"],
        "origin_warehouse": row["origin_warehouse"],
        "destination_store": row["destination_store"],
        "region": row["region"],
        "carrier": row["carrier"],
        "service_level": row["service_level"],
        "units": int(row["units"]),
        "distance_km": int(row["distance_km"]),
        "dispatched_at": pd.to_datetime(row["dispatch_ts"]).strftime("%Y-%m-%d %H:%M"),
        "last_checkpoint": row["last_checkpoint"],
        "last_scan_at": pd.to_datetime(row["last_scan_ts"]).strftime("%Y-%m-%d %H:%M"),
        "hours_since_last_scan": round(
            (dt.datetime.now() - pd.to_datetime(row["last_scan_ts"])).total_seconds() / 3600, 1
        ),
        "progress_pct": float(row["progress_pct"]),
        **_predict_arrival(row),
    })


@tool
def find_shipments(store_or_region: str = "", status: str = "", late_only: bool = False) -> str:
    """List live shipments, filtered by destination store, region or status.

    Use for "what is coming into Rouen", "what is late in Spain", "show me
    everything held at customs". Returns up to 25 shipments.
    """
    live = _load("live_shipments")
    needle = store_or_region.strip().upper()

    if needle:
        stores = _load("stores")
        matched_ids = stores[
            stores["store_name"].str.upper().str.contains(needle, na=False)
            | (stores["store_id"].str.upper() == needle)
            | stores["region"].str.upper().str.contains(needle, na=False)
        ]["store_id"].tolist()
        live = live[
            live["destination_store"].isin(matched_ids)
            | live["region"].str.upper().str.contains(needle, na=False)
        ]

    if status.strip():
        live = live[live["status"] == status.strip().lower()]

    if live.empty:
        return json.dumps({"matches": 0, "message": f"No live shipments for '{store_or_region}'."})

    rows = []
    for _, row in live.iterrows():
        prediction = _predict_arrival(row)
        if late_only and not prediction["projected_to_miss_promise"]:
            continue
        rows.append({
            "shipment_id": row["shipment_id"],
            "order_id": row["order_id"],
            "destination_store": row["destination_store"],
            "carrier": row["carrier"],
            "status": row["status"],
            "last_checkpoint": row["last_checkpoint"],
            "progress_pct": float(row["progress_pct"]),
            **{key: prediction[key] for key in
               ("promised_arrival", "predicted_arrival", "projected_delay_hours",
                "projected_to_miss_promise")},
        })

    rows.sort(key=lambda item: item["projected_delay_hours"], reverse=True)
    return json.dumps({"matches": len(rows), "shipments": rows[:25]})


@tool
def get_lane_performance(lane_or_carrier: str = "", worst: bool = False) -> str:
    """On-time rate, median and p90 transit per lane and carrier, last 90 days.

    Use for "how is carrier X performing", "which lanes are worst". Only lanes
    with at least 20 shipments in the window are treated as measurable.
    """
    performance = _load("lane_performance", CURATED_DIR)
    measurable = performance[performance["sufficient_volume"]]

    needle = lane_or_carrier.strip().upper()
    if needle:
        measurable = measurable[
            measurable["lane"].str.upper().str.contains(needle, na=False)
            | measurable["carrier"].str.upper().str.contains(needle, na=False)
        ]

    if measurable.empty:
        return json.dumps({
            "matches": 0,
            "message": f"No lane with at least 20 shipments matching '{lane_or_carrier}'. "
                       "Below that volume the on-time rate is not a measurement.",
        })

    ordered = measurable.sort_values("on_time_rate_pct", ascending=worst or True)
    return json.dumps({
        "matches": int(len(measurable)),
        "window": "last 90 days",
        "minimum_shipments": 20,
        "lanes": ordered.head(15).drop(columns=["sufficient_volume"]).to_dict(orient="records"),
    })


@tool
def get_store_cover(store_or_region: str = "", at_risk_only: bool = True) -> str:
    """Days of cover per store and category, with projected stock-out dates.

    Use for "is this store going to run out", "what is at risk in Poland".
    Includes units already in transit, per the replenishment policy.
    """
    cover = _load("store_cover", CURATED_DIR)
    needle = store_or_region.strip().upper()

    if needle:
        cover = cover[
            cover["store_name"].str.upper().str.contains(needle, na=False)
            | (cover["store_id"].str.upper() == needle)
            | cover["region"].str.upper().str.contains(needle, na=False)
        ]
    if at_risk_only:
        cover = cover[cover["risk"].isin(["critical", "high"])]

    if cover.empty:
        return json.dumps({
            "matches": 0,
            "message": "Nothing at risk for that filter."
            if at_risk_only else f"No store matching '{store_or_region}'.",
        })

    return json.dumps({
        "matches": int(len(cover)),
        "rows": cover.head(25).to_dict(orient="records"),
    })


@tool
def search_logistics_procedures(query: str) -> str:
    """Search the logistics procedures, policies and status definitions.

    Use for "what do I do when a delivery is late", "when can I claim against a
    carrier", "what does held at customs mean", "what is the cover target for
    footwear". Returns the matching sections.
    """
    from rag.doc_index import search

    hits = search(query, top_k=4)
    if not hits:
        return json.dumps({
            "matches": 0,
            "message": "Nothing in the procedures covers that. Do not answer from "
                       "general knowledge; escalate to Regional Logistics.",
        })
    return json.dumps({"matches": len(hits), "sections": hits})


ALL_TOOLS = [
    track_shipment,
    find_shipments,
    get_lane_performance,
    get_store_cover,
    search_logistics_procedures,
]
