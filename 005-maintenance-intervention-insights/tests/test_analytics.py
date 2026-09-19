"""Tests for the deterministic half of the pipeline.

The analytics layer is where every number in the management report comes from, so
it is the part that has to be right. None of these tests need an API key.

Run with:  pytest -q
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.analytics import (
    _downtime_by,
    _top_counts,
    month_over_month,
    repeat_offenders,
)
from schemas import InterventionRecord, strict_json_schema


def make_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["reported_at"] = pd.to_datetime(frame["reported_at"])
    frame["month"] = frame["reported_at"].dt.strftime("%Y-%m")
    frame["downtime_minutes"] = pd.to_numeric(frame["downtime_minutes"], errors="coerce")
    return frame


BASE_ROWS = [
    {
        "intervention_id": "A-1", "equipment_tag": "HRM-01", "reported_at": "2026-01-03",
        "failure_category": "hydraulic_leak", "root_cause_class": "wear_end_of_life",
        "resolution": "resolved", "downtime_minutes": 95,
    },
    {
        "intervention_id": "A-2", "equipment_tag": "HRM-01", "reported_at": "2026-01-14",
        "failure_category": "hydraulic_leak", "root_cause_class": "deferred_repair",
        "resolution": "temporary_fix", "downtime_minutes": 85,
    },
    {
        "intervention_id": "A-3", "equipment_tag": "HRM-01", "reported_at": "2026-02-02",
        "failure_category": "hydraulic_leak", "root_cause_class": "wear_end_of_life",
        "resolution": "resolved", "downtime_minutes": 240,
    },
    {
        "intervention_id": "A-4", "equipment_tag": "BF-02", "reported_at": "2026-02-11",
        "failure_category": "blockage_or_fouling", "root_cause_class": "upstream_contamination",
        "resolution": "resolved", "downtime_minutes": 60,
    },
]


def test_top_counts_returns_share_of_total():
    frame = make_frame(BASE_ROWS)
    counts = _top_counts(frame, "failure_category")

    assert counts[0]["value"] == "hydraulic_leak"
    assert counts[0]["count"] == 3
    assert counts[0]["share_pct"] == 75.0


def test_downtime_by_equipment_sorts_by_total_not_count():
    frame = make_frame(BASE_ROWS)
    result = _downtime_by(frame, "equipment_tag")

    assert result[0]["value"] == "HRM-01"
    assert result[0]["total_downtime_minutes"] == 420
    assert result[0]["interventions"] == 3


def test_missing_downtime_does_not_drag_the_average_down():
    # -1 is the sentinel the extraction stage writes when the report gave no
    # duration. Treating it as a real zero-ish value is the bug this guards.
    rows = BASE_ROWS + [
        {
            "intervention_id": "A-5", "equipment_tag": "CM-01", "reported_at": "2026-02-20",
            "failure_category": "overheating", "root_cause_class": "unknown",
            "resolution": "resolved", "downtime_minutes": -1,
        }
    ]
    frame = make_frame(rows)
    frame["downtime_minutes"] = frame["downtime_minutes"].replace(-1, pd.NA)
    frame["downtime_minutes"] = pd.to_numeric(frame["downtime_minutes"], errors="coerce")

    assert frame["downtime_minutes"].isna().sum() == 1
    assert frame["downtime_minutes"].mean() == pytest.approx(120.0)
    assert frame["downtime_minutes"].sum() == 480


def test_repeat_offenders_counts_across_months_not_within_one():
    # The point of the repeat-offender view is that it sees January and February
    # together. A month-scoped count would show two separate two-event problems
    # and rank neither as worth acting on.
    frame = make_frame(BASE_ROWS)
    offenders = repeat_offenders(frame, min_events=3)

    assert len(offenders) == 1
    assert offenders[0]["equipment_tag"] == "HRM-01"
    assert offenders[0]["events"] == 3
    assert offenders[0]["still_open_or_temporary"] == 1
    assert "deferred_repair" in offenders[0]["root_cause_classes"]


def test_repeat_offenders_respects_the_threshold():
    frame = make_frame(BASE_ROWS)
    assert repeat_offenders(frame, min_events=4) == []
    assert len(repeat_offenders(frame, min_events=2)) == 1


def test_month_over_month_computes_deltas_and_category_changes():
    frame = make_frame(BASE_ROWS)
    comparison = month_over_month(frame, "2026-02")

    assert comparison["previous_month"] == "2026-01"
    assert comparison["interventions"]["current"] == 2
    assert comparison["interventions"]["previous"] == 2
    assert comparison["interventions"]["change_pct"] == 0.0
    assert comparison["downtime_minutes"]["current"] == 300
    assert comparison["downtime_minutes"]["previous"] == 180
    assert comparison["downtime_minutes"]["change_pct"] == pytest.approx(66.7)
    assert "blockage_or_fouling" in comparison["new_failure_categories"]


def test_month_over_month_handles_the_first_month():
    frame = make_frame(BASE_ROWS)
    comparison = month_over_month(frame, "2026-01")

    assert comparison["previous_month"] is None
    assert "note" in comparison


def test_strict_schema_is_valid_for_openai_structured_outputs():
    schema = strict_json_schema(InterventionRecord)

    # Strict mode rejects a schema unless every object forbids extra properties
    # and lists every property as required.
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for item in node:
                check(item)

    check(schema)
    assert set(schema["required"]) == set(schema["properties"])
    # Must survive a round trip as JSON, since that is how it reaches the API.
    json.dumps(schema)


def test_schema_exposes_the_closed_vocabularies():
    schema = strict_json_schema(InterventionRecord)
    enum_names = set(schema["$defs"].keys())

    # The report groups by these, so they must reach the model as enums rather
    # than as free-text fields.
    assert {"MachineType", "FailureCategory", "RootCauseClass", "Resolution", "Severity"} <= enum_names
    assert "temporary_fix" in schema["$defs"]["Resolution"]["enum"]
