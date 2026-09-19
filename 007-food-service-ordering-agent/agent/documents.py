"""Purchase order PDF generation and delivery.

Email defaults to dry-run, writing an .eml file to outbox/ instead of opening an
SMTP connection. An agent that can send real email to a real supplier on the
strength of a sentence typed into a chat box needs that default to be off by
deliberate choice, not by forgetting to configure it.
"""
from __future__ import annotations

import datetime as dt
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTBOX = BASE_DIR / "outbox"

SITE_NAME = os.getenv("SITE_NAME", "Staff Restaurant - Site 01")
SITE_ADDRESS = os.getenv("SITE_ADDRESS", "12 rue de l'Industrie, 69100 Villeurbanne, France")
SITE_EMAIL = os.getenv("SITE_EMAIL", "site01@foodservice.example")


def _dry_run() -> bool:
    return os.getenv("EMAIL_DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "on"}


def build_po_pdf(
    po_number: str,
    supplier: dict,
    lines: list[dict],
    total_eur: float,
    requested_delivery_date: dt.date | None = None,
    notes: str = "",
) -> Path:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    path = OUTBOX / f"{po_number}.pdf"

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=f"Purchase Order {po_number}",
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>PURCHASE ORDER {po_number}</b>", styles["Title"]))
    story.append(Spacer(1, 6 * mm))

    header = [
        ["From", "To"],
        [
            f"{SITE_NAME}<br/>{SITE_ADDRESS}<br/>{SITE_EMAIL}",
            f"{supplier['name']}<br/>{supplier.get('contact_name', '')}<br/>"
            f"{supplier.get('email', '')}<br/>Account: {supplier.get('account_number', '-')}",
        ],
    ]
    header_table = Table(
        [[Paragraph(cell, styles["BodyText"]) for cell in row] for row in header],
        colWidths=[85 * mm, 85 * mm],
    )
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 5 * mm))

    meta = [
        f"<b>Order date:</b> {dt.date.today().isoformat()}",
        f"<b>Requested delivery:</b> "
        f"{requested_delivery_date.isoformat() if requested_delivery_date else 'Next scheduled delivery'}",
        f"<b>Ordering channel:</b> automated agent, confirmed by site manager",
    ]
    for line in meta:
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 5 * mm))

    table_data = [["SKU", "Product", "Qty", "Unit", "Cases", "Case price", "Line total"]]
    for line in lines:
        table_data.append([
            line["sku"],
            line["product"],
            f"{line['quantity']:g}",
            line["unit"],
            str(line.get("cases", "-")),
            f"{line['case_price_eur']:.2f} EUR",
            f"{line['line_total_eur']:.2f} EUR",
        ])
    table_data.append(["", "", "", "", "", "TOTAL", f"{total_eur:.2f} EUR"])

    lines_table = Table(
        table_data,
        colWidths=[22 * mm, 52 * mm, 15 * mm, 15 * mm, 15 * mm, 24 * mm, 27 * mm],
    )
    lines_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -2), 0.4, colors.HexColor("#B0BEC5")),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (5, -1), (-1, -1), 0.8, colors.black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F5F7FA")]),
        ])
    )
    story.append(lines_table)

    if notes:
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph(f"<b>Notes:</b> {notes}", styles["BodyText"]))

    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            "<font size=7 color='#607D8B'>Generated automatically from the site "
            "inventory forecast. Please confirm receipt by reply.</font>",
            styles["BodyText"],
        )
    )

    document.build(story)
    return path


def send_po_email(
    po_number: str,
    supplier: dict,
    pdf_path: Path,
    lines: list[dict],
    total_eur: float,
) -> dict:
    """Send the PO, or write it to the outbox when EMAIL_DRY_RUN is on."""
    summary = "\n".join(
        f"  - {line['product']}: {line['quantity']:g} {line['unit']} "
        f"({line.get('cases', '-')} case(s))"
        for line in lines
    )
    body = (
        f"Dear {supplier.get('contact_name') or supplier['name']},\n\n"
        f"Please find attached purchase order {po_number} from {SITE_NAME}.\n\n"
        f"Order summary:\n{summary}\n\n"
        f"Order total: {total_eur:.2f} EUR\n\n"
        f"Please confirm receipt and the expected delivery date by reply.\n\n"
        f"Kind regards,\n{SITE_NAME}\n{SITE_EMAIL}\n"
    )

    message = EmailMessage()
    message["Subject"] = f"Purchase Order {po_number} - {SITE_NAME}"
    message["From"] = SITE_EMAIL
    message["To"] = supplier["email"]
    message.set_content(body)
    message.add_attachment(
        pdf_path.read_bytes(),
        maintype="application",
        subtype="pdf",
        filename=pdf_path.name,
    )

    if _dry_run():
        OUTBOX.mkdir(parents=True, exist_ok=True)
        eml_path = OUTBOX / f"{po_number}.eml"
        eml_path.write_bytes(bytes(message))
        return {
            "sent": False,
            "dry_run": True,
            "to": supplier["email"],
            "eml_path": str(eml_path),
            "message": (
                f"Dry run: purchase order written to {eml_path.name} instead of "
                f"being emailed. Set EMAIL_DRY_RUN=false to send for real."
            ),
        }

    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as server:
        if os.getenv("SMTP_STARTTLS", "true").lower() == "true":
            server.starttls()
        if os.getenv("SMTP_USERNAME"):
            server.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        server.send_message(message)

    return {
        "sent": True,
        "dry_run": False,
        "to": supplier["email"],
        "message": f"Purchase order {po_number} emailed to {supplier['email']}.",
    }
