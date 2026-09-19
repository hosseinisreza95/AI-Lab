"""Stage 1 - turn a free-text intervention report into a structured record.

The LLM does three jobs at once here: it repairs typos and expands shop-floor
shorthand, it extracts the facts, and it maps them onto a closed vocabulary. The
closed vocabulary is the important part. Free-form category strings would give
the reporting stage forty near-duplicate labels to group by, which is exactly the
state the original free text was already in.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from database.models import Intervention, SessionLocal, init_db
from schemas import InterventionRecord, strict_json_schema

load_dotenv()

MODEL = os.getenv("STRUCTURING_MODEL", "gpt-4o-mini")
CONFIDENCE_FLOOR = float(os.getenv("CONFIDENCE_FLOOR", "0.6"))

SYSTEM_PROMPT = """You convert free-text maintenance intervention reports from a \
steel and iron plant into structured records.

The reports are written on a tablet by technicians on a shop floor, immediately \
after an intervention. They contain typos, missing punctuation, inconsistent \
capitalisation, and heavy abbreviation: hyd = hydraulic, cyl = cylinder, \
brg = bearing, TC = thermocouple, LS = limit switch, ckt = circuit, \
HPU = hydraulic power unit, WO = work order, qtr = quarter, dp = differential \
pressure, LVDT = linear position sensor, VFD = variable frequency drive, \
seg = segment, assy = assembly, eng = engineering.

Rules:
- Extract only what the report states. Never infer a downtime figure, a part \
number or a cause that is not there. Use -1 for unstated downtime and an empty \
string for an unstated cause.
- Normalise terminology. The same component must get the same name every time, \
in lowercase standard English, regardless of how the technician spelled it.
- Distinguish the symptom from the cause. "Seal blew" is the symptom; "rod is \
scored so seals keep failing" is the cause. root_cause_detail holds the cause.
- resolution has three states. Read the report carefully: a technician who wrote \
"temp fix" or "NOT RESOLVED" has told you which one it is, even if they also \
describe work they completed.
- recurring is true only when the report itself signals repetition: "again", \
"3rd time", "same as last month", "keeps happening".
- planned_work is true for shutdown or maintenance-window work, which the report \
usually marks with the word "planned" or a long duration during a shutdown.
- escalation_needed is true when the technician asks for engineering, a design \
change, or work outside their own scope.
- Set extraction_confidence below 0.6 when the report is too ambiguous to \
structure reliably. That is a useful answer, not a failure."""

USER_TEMPLATE = """Equipment tag: {equipment_tag}
Line: {line}
Plant: {plant}
Alert code: {alert_code}
Reported at: {reported_at}

REPORT TEXT:
{raw_text}"""


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def structure_one(client: OpenAI, row: dict) -> InterventionRecord | None:
    """Extract one record. Returns None when the model output cannot be validated."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(**row)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "intervention_record",
                "strict": True,
                "schema": strict_json_schema(InterventionRecord),
            },
        },
    )

    payload = response.choices[0].message.content
    try:
        return InterventionRecord.model_validate_json(payload)
    except ValidationError as exc:
        # Strict mode makes this rare, but a schema change deployed ahead of a
        # prompt change would land here, and silently dropping records would
        # corrupt the monthly counts without anyone noticing.
        print(f"  ! validation failed for {row['intervention_id']}: {exc.error_count()} errors")
        return None


def run(csv_path: str, limit: int | None = None, skip_existing: bool = True) -> dict:
    init_db()
    client = _client()
    frame = pd.read_csv(csv_path)
    if limit:
        frame = frame.head(limit)

    session = SessionLocal()
    stats = {"total": len(frame), "structured": 0, "skipped": 0, "failed": 0, "low_confidence": 0}

    try:
        existing = {
            row[0] for row in session.query(Intervention.intervention_id).all()
        } if skip_existing else set()

        for _, row in frame.iterrows():
            payload = row.to_dict()
            if payload["intervention_id"] in existing:
                stats["skipped"] += 1
                continue

            record = structure_one(client, payload)
            if record is None:
                stats["failed"] += 1
                continue

            if record.extraction_confidence < CONFIDENCE_FLOOR:
                stats["low_confidence"] += 1

            session.add(
                Intervention(
                    intervention_id=payload["intervention_id"],
                    plant=payload["plant"],
                    line=payload["line"],
                    equipment_tag=payload["equipment_tag"],
                    reported_at=datetime.fromisoformat(payload["reported_at"]),
                    technician_id=payload["technician_id"],
                    alert_code=payload["alert_code"],
                    raw_text=payload["raw_text"],
                    summary=record.summary,
                    machine_type=record.machine_type.value,
                    component=record.component,
                    failure_category=record.failure_category.value,
                    root_cause_class=record.root_cause_class.value,
                    root_cause_detail=record.root_cause_detail,
                    action_taken=record.action_taken,
                    resolution=record.resolution.value,
                    severity=record.severity.value,
                    downtime_minutes=record.downtime_minutes,
                    planned_work=record.planned_work,
                    parts_replaced=json.dumps(record.parts_replaced),
                    recurring=record.recurring,
                    recommendation=record.recommendation,
                    escalation_needed=record.escalation_needed,
                    extraction_confidence=record.extraction_confidence,
                    structuring_model=MODEL,
                )
            )
            stats["structured"] += 1
            print(
                f"  [{stats['structured']:>3}] {payload['intervention_id']} -> "
                f"{record.failure_category.value} / {record.root_cause_class.value} "
                f"({record.extraction_confidence:.2f})"
            )

        session.commit()
    finally:
        session.close()

    return stats
