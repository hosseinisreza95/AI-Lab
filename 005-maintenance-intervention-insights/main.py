"""CLI for the two-stage intervention insights pipeline.

    python main.py structure                 # stage 1: free text -> structured DB
    python main.py analytics --month 2026-02 # inspect the computed numbers
    python main.py report --month 2026-02    # stage 2: analytics -> monthly report
    python main.py report --all              # one report per month in the data
    python main.py inspect INT-2026-0109     # see one record before and after
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "raw_interventions.csv"


def cmd_structure(args: argparse.Namespace) -> int:
    from pipeline.structuring import run

    print(f"Structuring interventions from {args.csv} ...")
    stats = run(str(args.csv), limit=args.limit, skip_existing=not args.force)
    print("\n" + json.dumps(stats, indent=2))
    if stats["low_confidence"]:
        print(
            f"\n{stats['low_confidence']} record(s) below the confidence floor. "
            "They are stored and flagged, not dropped."
        )
    return 0


def cmd_analytics(args: argparse.Namespace) -> int:
    from pipeline.analytics import available_months, build_analytics

    months = available_months()
    if not months:
        print("No structured records. Run `python main.py structure` first.")
        return 1

    month = args.month or months[-1]
    print(json.dumps(build_analytics(month), indent=2, default=str))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from pipeline.analytics import available_months
    from pipeline.reporting import generate_report

    months = available_months()
    if not months:
        print("No structured records. Run `python main.py structure` first.")
        return 1

    targets = months if args.all else [args.month or months[-1]]
    for month in targets:
        print(f"\nGenerating report for {month} ...")
        result = generate_report(month)
        print(result["markdown"])
        if result["path"]:
            print(f"\nSaved to {result['path']}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from database.models import Intervention, SessionLocal

    session = SessionLocal()
    try:
        record = (
            session.query(Intervention)
            .filter(Intervention.intervention_id == args.intervention_id)
            .first()
        )
        if not record:
            print(f"No record for {args.intervention_id}.")
            return 1

        print("=" * 72)
        print("RAW (as written on the tablet)")
        print("=" * 72)
        print(record.raw_text)
        print()
        print("=" * 72)
        print("STRUCTURED")
        print("=" * 72)
        print(json.dumps(
            {
                "summary": record.summary,
                "machine_type": record.machine_type,
                "component": record.component,
                "failure_category": record.failure_category,
                "root_cause_class": record.root_cause_class,
                "root_cause_detail": record.root_cause_detail,
                "action_taken": record.action_taken,
                "resolution": record.resolution,
                "severity": record.severity,
                "downtime_minutes": record.downtime_minutes,
                "planned_work": record.planned_work,
                "parts_replaced": record.parts,
                "recurring": record.recurring,
                "recommendation": record.recommendation,
                "escalation_needed": record.escalation_needed,
                "extraction_confidence": record.extraction_confidence,
            },
            indent=2,
        ))
    finally:
        session.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Industrial maintenance intervention insights pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    structure = subparsers.add_parser("structure", help="Stage 1: structure free-text reports")
    structure.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    structure.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    structure.add_argument("--force", action="store_true", help="Re-process records already in the DB")
    structure.set_defaults(func=cmd_structure)

    analytics = subparsers.add_parser("analytics", help="Print the computed analytics object")
    analytics.add_argument("--month", type=str, default=None, help="YYYY-MM")
    analytics.set_defaults(func=cmd_analytics)

    report = subparsers.add_parser("report", help="Stage 2: generate the monthly report")
    report.add_argument("--month", type=str, default=None, help="YYYY-MM")
    report.add_argument("--all", action="store_true", help="Generate a report for every month")
    report.set_defaults(func=cmd_report)

    inspect = subparsers.add_parser("inspect", help="Show one record raw and structured")
    inspect.add_argument("intervention_id", type=str)
    inspect.set_defaults(func=cmd_inspect)

    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    sys.exit(parsed.func(parsed))
