"""The contract between the messy shop floor and the reporting layer.

Everything downstream of stage 1 reads these fields and nothing else. Getting the
taxonomy right here is what makes the monthly report possible: if the LLM is free
to invent category names, two spellings of the same failure become two trends.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MachineType(str, Enum):
    """Closed vocabulary, because the report groups by this field.

    'other' exists on purpose. A model forced to pick from a closed list with no
    escape hatch will mislabel rather than admit uncertainty, and a mislabelled
    record is harder to find than an unlabelled one.
    """

    ROLLING_MILL = "rolling_mill"
    BLAST_FURNACE = "blast_furnace"
    SINTER_PLANT = "sinter_plant"
    CONTINUOUS_CASTER = "continuous_caster"
    CONVEYOR = "conveyor"
    HYDRAULIC_POWER_UNIT = "hydraulic_power_unit"
    FAN_OR_BLOWER = "fan_or_blower"
    GEARBOX = "gearbox"
    ELECTRICAL_DRIVE = "electrical_drive"
    GAS_CLEANING = "gas_cleaning"
    OTHER = "other"


class FailureCategory(str, Enum):
    MECHANICAL_WEAR = "mechanical_wear"
    HYDRAULIC_LEAK = "hydraulic_leak"
    ELECTRICAL_FAULT = "electrical_fault"
    INSTRUMENT_OR_SENSOR = "instrument_or_sensor"
    LUBRICATION = "lubrication"
    BLOCKAGE_OR_FOULING = "blockage_or_fouling"
    OVERHEATING = "overheating"
    BEARING_FAILURE = "bearing_failure"
    STRUCTURAL_OR_FOUNDATION = "structural_or_foundation"
    CONTROL_OR_SOFTWARE = "control_or_software"
    OTHER = "other"


class RootCauseClass(str, Enum):
    """Why it failed, as opposed to what failed.

    This is the field management actually acts on. 'The seal blew' is a symptom;
    'the cylinder rod is scored so seals keep failing' is a budget decision.
    """

    WEAR_END_OF_LIFE = "wear_end_of_life"
    DESIGN_OR_INSTALLATION = "design_or_installation"
    UPSTREAM_CONTAMINATION = "upstream_contamination"
    MAINTENANCE_INTERVAL_TOO_LONG = "maintenance_interval_too_long"
    DEFERRED_REPAIR = "deferred_repair"
    MISALIGNMENT_OR_LOOSENESS = "misalignment_or_looseness"
    LOSS_OF_LUBRICATION = "loss_of_lubrication"
    ENVIRONMENTAL_INGRESS = "environmental_ingress"
    HUMAN_ERROR = "human_error"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Resolution(str, Enum):
    """Three states, not a boolean.

    A technician writing "tensioned as temp fix, NOT RESOLVED - needs re-lagging"
    has neither fixed it nor failed to fix it. Collapsing that into resolved=False
    loses the fact that the line is running, and collapsing it into True loses the
    outstanding work. Both readings produce a wrong monthly report.
    """

    RESOLVED = "resolved"
    TEMPORARY_FIX = "temporary_fix"
    NOT_RESOLVED = "not_resolved"


class InterventionRecord(BaseModel):
    """One structured maintenance intervention, extracted from free text."""

    summary: str = Field(
        ...,
        description="One clear sentence in standard technical English describing what "
                    "happened and what was done. No typos, no shorthand.",
    )
    machine_type: MachineType
    component: str = Field(
        ...,
        description="The specific component that failed, normalised to lowercase "
                    "standard terminology, e.g. 'hydraulic cylinder rod seal', "
                    "'thermocouple', 'drive coupling element'.",
    )
    failure_category: FailureCategory
    root_cause_class: RootCauseClass
    root_cause_detail: str = Field(
        ...,
        description="Short phrase giving the specific cause, or an empty string if "
                    "the report does not state one. Never guess.",
    )
    action_taken: str = Field(..., description="What the technician actually did.")
    resolution: Resolution
    severity: Severity
    downtime_minutes: int = Field(
        ...,
        description="Downtime in minutes as stated in the report. Use -1 if the "
                    "report does not state a duration. Do not estimate.",
    )
    planned_work: bool = Field(
        ...,
        description="True if this was planned shutdown or window work rather than a "
                    "breakdown response.",
    )
    parts_replaced: list[str] = Field(
        ...,
        description="Normalised part names replaced, or an empty list.",
    )
    recurring: bool = Field(
        ...,
        description="True only if the report itself indicates the problem has "
                    "happened before on this equipment (e.g. 'again', '3rd time').",
    )
    recommendation: str = Field(
        ...,
        description="The technician's recommendation for next time, rewritten "
                    "clearly. Empty string if none was given.",
    )
    escalation_needed: bool = Field(
        ...,
        description="True if the report asks for engineering action, a design change, "
                    "or work beyond the technician's scope.",
    )
    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident you are in this extraction. Below 0.6 means the "
                    "source text was too ambiguous to structure reliably.",
    )


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into an OpenAI strict structured-output schema.

    Strict mode requires every property to be listed in `required` and
    `additionalProperties: false` on every object. Optional fields are therefore
    avoided in the model above in favour of sentinel values (-1, empty string),
    which keeps this conversion a simple walk rather than a rewrite of `anyOf`
    unions.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def harden(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            # Descriptions are instructions to the model; titles are noise.
            node.pop("title", None)
            return {key: harden(value) for key, value in node.items()}
        if isinstance(node, list):
            return [harden(item) for item in node]
        return node

    hardened = harden(schema)
    if definitions:
        hardened["$defs"] = harden(definitions)
    return hardened
