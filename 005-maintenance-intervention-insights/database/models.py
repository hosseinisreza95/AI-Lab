"""Internal database for structured intervention records.

SQLite keeps the experiment self-contained. The schema is deliberately the same
shape a plant historian or CMMS table would be, so the analytics layer would not
change if this were swapped for the real system.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
# Overridable so tests can point at a throwaway file instead of the real database.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'interventions.db'}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Intervention(Base):
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True, index=True)

    # --- Fields carried over unchanged from the source system ---------------
    intervention_id = Column(String, unique=True, index=True)
    plant = Column(String, index=True)
    line = Column(String)
    equipment_tag = Column(String, index=True)
    reported_at = Column(DateTime, index=True)
    technician_id = Column(String)
    alert_code = Column(String)
    raw_text = Column(Text)

    # --- Fields produced by the LLM structuring stage ------------------------
    summary = Column(Text)
    machine_type = Column(String, index=True)
    component = Column(String, index=True)
    failure_category = Column(String, index=True)
    root_cause_class = Column(String, index=True)
    root_cause_detail = Column(Text)
    action_taken = Column(Text)
    resolution = Column(String, index=True)
    severity = Column(String)
    downtime_minutes = Column(Integer)
    planned_work = Column(Boolean)
    parts_replaced = Column(Text)  # JSON-encoded list
    recurring = Column(Boolean)
    recommendation = Column(Text)
    escalation_needed = Column(Boolean)
    extraction_confidence = Column(Float)

    # --- Provenance ----------------------------------------------------------
    structured_at = Column(DateTime, default=dt.datetime.now)
    structuring_model = Column(String)

    @property
    def parts(self) -> list[str]:
        return json.loads(self.parts_replaced or "[]")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
