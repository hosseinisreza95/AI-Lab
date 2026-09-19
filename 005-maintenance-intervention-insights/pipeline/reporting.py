"""Stage 2 - write the monthly management report from the computed analytics.

The model receives finished arithmetic and writes prose around it. It is told, in
as many words, that it may not compute anything, because the failure mode this
guards against is not a hallucinated fact but a recomputed total that is slightly
wrong and completely plausible.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.analytics import build_analytics

load_dotenv()

MODEL = os.getenv("REPORTING_MODEL", "gpt-4o-mini")
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

SYSTEM_PROMPT = """You write the monthly maintenance report for the management team \
of a steel and iron production site. Your readers decide where to spend the \
maintenance budget. They are not engineers and they will not read past the first \
paragraph unless it tells them something they can act on.

You are given a JSON analytics object. Every number in it has already been \
computed from the structured intervention database.

Hard rules:
- Use only numbers that appear in the JSON. Do not add, average, convert or \
recompute anything. If you want a figure that is not there, leave it out.
- Convert minutes to hours only when the JSON value is clearly large, and show \
your source figure alongside, e.g. "1,240 minutes (about 21 hours)".
- Do not soften the findings. If one piece of equipment caused a third of the \
downtime, say which and say so in the first paragraph.
- Where repeat_offenders shows the same equipment and failure category \
recurring, treat it as the main story of the month.
- Where open_or_temporary contains items, they are carried risk. List them.
- Where technician_recommendations repeat the same request, that is a signal that \
a decision has been deferred. Say it plainly, without blaming anyone.
- If low_confidence_extractions is above zero, add a one-line data-quality note \
at the end so the reader knows the coverage is not perfect.

Write in this structure, in Markdown:

# Maintenance Report - {month}

## Headline
Three or four sentences. What happened this month, how it compares to last \
month, and the single thing most worth acting on.

## Where the Downtime Went
A short table and two or three sentences of interpretation.

## Recurring Failures
The repeat offenders, with what is actually causing them.

## Carried Risk
Open and temporary fixes still outstanding.

## Recommended Actions
A numbered list of three to five concrete actions, each tied to the evidence \
that supports it. Order them by the downtime they would remove.

## Data Quality
One or two lines. Only if there is something to report."""


def generate_report(month: str, analytics: dict | None = None, save: bool = True) -> dict:
    analytics = analytics or build_analytics(month)

    if analytics.get("interventions") == 0:
        return {"month": month, "markdown": f"# Maintenance Report - {month}\n\nNo records.",
                "analytics": analytics, "path": None}

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=2500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.replace("{month}", month)},
            {
                "role": "user",
                "content": (
                    "Here is the analytics object for the month. Write the report.\n\n"
                    f"{json.dumps(analytics, indent=2, default=str)}"
                ),
            },
        ],
    )
    markdown = response.choices[0].message.content or ""

    path = None
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"maintenance-report-{month}.md"
        footer = (
            f"\n\n---\n\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"from {analytics['headline']['interventions']} structured intervention "
            f"records. Model: {MODEL}. All figures computed in the analytics layer, "
            f"not by the language model.*\n"
        )
        path.write_text(markdown + footer, encoding="utf-8")

        # Keeping the analytics next to the report is what makes a number in the
        # prose checkable six months later.
        (REPORTS_DIR / f"analytics-{month}.json").write_text(
            json.dumps(analytics, indent=2, default=str), encoding="utf-8"
        )

    return {"month": month, "markdown": markdown, "analytics": analytics, "path": path}
