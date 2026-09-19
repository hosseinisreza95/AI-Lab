"""Deterministic analytics over the structured records.

Every number in the monthly report is computed here, in pandas, and handed to the
report writer as finished arithmetic. The LLM never counts anything. That split
is the whole reason the report can be trusted: a language model asked to total
downtime across fifty records will produce a plausible number, and plausible is
not the same as correct.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from database.models import Intervention, SessionLocal

# Columns the report writer is allowed to see.
FRAME_COLUMNS = [
    "intervention_id", "plant", "line", "equipment_tag", "reported_at",
    "machine_type", "component", "failure_category", "root_cause_class",
    "root_cause_detail", "resolution", "severity", "downtime_minutes",
    "planned_work", "parts_replaced", "recurring", "recommendation",
    "escalation_needed", "extraction_confidence", "summary",
]


def load_frame(month: str | None = None) -> pd.DataFrame:
    """Load structured records, optionally filtered to a YYYY-MM month."""
    session = SessionLocal()
    try:
        rows = session.query(Intervention).all()
        frame = pd.DataFrame(
            [{column: getattr(row, column) for column in FRAME_COLUMNS} for row in rows]
        )
    finally:
        session.close()

    if frame.empty:
        return frame

    frame["reported_at"] = pd.to_datetime(frame["reported_at"])
    frame["month"] = frame["reported_at"].dt.strftime("%Y-%m")
    frame["parts_replaced"] = frame["parts_replaced"].apply(
        lambda value: json.loads(value or "[]")
    )
    # -1 is the sentinel for "the report did not say". Averaging it in would drag
    # every mean downwards, so it becomes NaN and is excluded from statistics.
    frame["downtime_minutes"] = frame["downtime_minutes"].replace(-1, pd.NA)
    frame["downtime_minutes"] = pd.to_numeric(frame["downtime_minutes"], errors="coerce")

    if month:
        frame = frame[frame["month"] == month]
    return frame


def _top_counts(frame: pd.DataFrame, column: str, limit: int = 5) -> list[dict]:
    if frame.empty:
        return []
    counts = frame[column].value_counts().head(limit)
    total = len(frame)
    return [
        {
            "value": str(value),
            "count": int(count),
            "share_pct": round(count / total * 100, 1),
        }
        for value, count in counts.items()
    ]


def _downtime_by(frame: pd.DataFrame, column: str, limit: int = 5) -> list[dict]:
    if frame.empty:
        return []
    grouped = (
        frame.groupby(column)["downtime_minutes"]
        .agg(["sum", "mean", "count"])
        .sort_values("sum", ascending=False)
        .head(limit)
    )
    return [
        {
            "value": str(index),
            "total_downtime_minutes": int(row["sum"]) if pd.notna(row["sum"]) else 0,
            "avg_downtime_minutes": round(float(row["mean"]), 1) if pd.notna(row["mean"]) else None,
            "interventions": int(row["count"]),
        }
        for index, row in grouped.iterrows()
    ]


def repeat_offenders(frame: pd.DataFrame, min_events: int = 3) -> list[dict]:
    """Equipment + failure category pairs that keep coming back.

    This is the single most useful output for management, and it is the one thing
    that was impossible to produce from the free text.
    """
    if frame.empty:
        return []
    grouped = frame.groupby(["equipment_tag", "failure_category"]).agg(
        events=("intervention_id", "count"),
        total_downtime=("downtime_minutes", "sum"),
        unresolved=("resolution", lambda values: int((values != "resolved").sum())),
        root_causes=("root_cause_class", lambda values: sorted(set(values))),
    )
    grouped = grouped[grouped["events"] >= min_events].sort_values(
        ["events", "total_downtime"], ascending=False
    )
    return [
        {
            "equipment_tag": index[0],
            "failure_category": index[1],
            "events": int(row["events"]),
            "total_downtime_minutes": int(row["total_downtime"]) if pd.notna(row["total_downtime"]) else 0,
            "still_open_or_temporary": int(row["unresolved"]),
            "root_cause_classes": row["root_causes"],
        }
        for index, row in grouped.iterrows()
    ]


def month_over_month(all_months: pd.DataFrame, month: str) -> dict:
    """Compare this month against the previous one, where a previous one exists."""
    if all_months.empty or "month" not in all_months:
        return {}

    months = sorted(all_months["month"].unique())
    if month not in months:
        return {}
    index = months.index(month)
    if index == 0:
        return {"previous_month": None, "note": "No prior month in the dataset."}

    previous = months[index - 1]
    current_frame = all_months[all_months["month"] == month]
    previous_frame = all_months[all_months["month"] == previous]

    def delta(current: float, prior: float) -> float | None:
        if not prior:
            return None
        return round((current - prior) / prior * 100, 1)

    current_count, previous_count = len(current_frame), len(previous_frame)
    current_downtime = float(current_frame["downtime_minutes"].sum())
    previous_downtime = float(previous_frame["downtime_minutes"].sum())

    return {
        "previous_month": previous,
        "interventions": {
            "current": current_count,
            "previous": previous_count,
            "change_pct": delta(current_count, previous_count),
        },
        "downtime_minutes": {
            "current": int(current_downtime),
            "previous": int(previous_downtime),
            "change_pct": delta(current_downtime, previous_downtime),
        },
        "new_failure_categories": sorted(
            set(current_frame["failure_category"]) - set(previous_frame["failure_category"])
        ),
        "resolved_failure_categories": sorted(
            set(previous_frame["failure_category"]) - set(current_frame["failure_category"])
        ),
    }


def build_analytics(month: str) -> dict:
    """Everything the report writer is given. No other source of numbers."""
    all_months = load_frame(None)
    frame = all_months[all_months["month"] == month] if not all_months.empty else all_months

    if frame.empty:
        return {"month": month, "interventions": 0, "note": "No records for this month."}

    unplanned = frame[~frame["planned_work"].astype(bool)]
    open_items = frame[frame["resolution"] != "resolved"]

    parts = [part for row in frame["parts_replaced"] for part in row]
    parts_counts = pd.Series(parts).value_counts().head(8) if parts else pd.Series(dtype=int)

    return {
        "month": month,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "headline": {
            "interventions": len(frame),
            "unplanned_interventions": len(unplanned),
            "planned_interventions": int(frame["planned_work"].astype(bool).sum()),
            "total_downtime_minutes": int(frame["downtime_minutes"].sum()),
            "unplanned_downtime_minutes": int(unplanned["downtime_minutes"].sum()),
            "avg_downtime_minutes": round(float(frame["downtime_minutes"].mean()), 1),
            "interventions_missing_downtime": int(frame["downtime_minutes"].isna().sum()),
            "recurring_share_pct": round(
                float(frame["recurring"].astype(bool).mean()) * 100, 1
            ),
            "escalations_requested": int(frame["escalation_needed"].astype(bool).sum()),
            "low_confidence_extractions": int((frame["extraction_confidence"] < 0.6).sum()),
        },
        "top_failure_categories": _top_counts(frame, "failure_category"),
        "top_root_cause_classes": _top_counts(frame, "root_cause_class"),
        "top_machine_types": _top_counts(frame, "machine_type"),
        "top_components": _top_counts(frame, "component", limit=8),
        "downtime_by_equipment": _downtime_by(frame, "equipment_tag"),
        "downtime_by_failure_category": _downtime_by(frame, "failure_category"),
        "resolution_breakdown": _top_counts(frame, "resolution"),
        "severity_breakdown": _top_counts(frame, "severity"),
        "repeat_offenders": repeat_offenders(all_months, min_events=3),
        "open_or_temporary": [
            {
                "intervention_id": row["intervention_id"],
                "equipment_tag": row["equipment_tag"],
                "resolution": row["resolution"],
                "summary": row["summary"],
                "recommendation": row["recommendation"],
            }
            for _, row in open_items.iterrows()
        ],
        "technician_recommendations": [
            {
                "equipment_tag": row["equipment_tag"],
                "recommendation": row["recommendation"],
                "escalation_needed": bool(row["escalation_needed"]),
            }
            for _, row in frame.iterrows()
            if isinstance(row["recommendation"], str) and row["recommendation"].strip()
        ],
        "parts_replaced": [
            {"part": str(part), "count": int(count)} for part, count in parts_counts.items()
        ],
        "month_over_month": month_over_month(all_months, month),
    }


def available_months() -> list[str]:
    frame = load_frame(None)
    return sorted(frame["month"].unique()) if not frame.empty else []
