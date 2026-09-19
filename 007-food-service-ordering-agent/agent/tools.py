"""Tools available to the ordering agent.

Two groups with very different risk profiles:

- Read tools (inventory, forecast, alerts, suppliers) are safe to call freely.
- Write tools create a purchase order and send it to a supplier. `create_purchase_order`
  only ever produces a draft; `send_purchase_order` is the single point where
  anything leaves the building, and it requires the PO number the draft returned.

That split is what makes a confirmation step possible at all. If drafting and
sending were one tool, the agent would have committed before the manager saw the
order.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid

from langchain_core.tools import tool

from agent.documents import build_po_pdf, send_po_email
from database.models import Product, PurchaseOrder, SessionLocal, Stock, Supplier
from forecasting.inventory import build_alerts
from forecasting.model import backtest, forecast_covers


def _supplier_dict(supplier: Supplier) -> dict:
    return {
        "id": supplier.id,
        "name": supplier.name,
        "contact_name": supplier.contact_name,
        "email": supplier.email,
        "phone": supplier.phone,
        "categories": supplier.categories,
        "order_cutoff_time": supplier.order_cutoff_time,
        "delivery_days": supplier.delivery_days,
        "account_number": supplier.account_number,
    }


@tool
def get_traffic_forecast(days: int = 7) -> str:
    """Forecast expected customer covers for the next N days.

    Use for questions about how busy the restaurant will be: tomorrow, this week,
    next week. Closed days are returned with zero covers and the reason.
    Returns a JSON list of dates with predicted covers and an 80% interval.
    """
    try:
        points = forecast_covers(horizon_days=max(1, min(days, 60)))
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(
        [
            {
                "date": point.service_date.isoformat(),
                "weekday": point.service_date.strftime("%A"),
                "predicted_covers": point.predicted_covers,
                "range": [point.lower, point.upper],
                "closed": point.is_closed,
                "note": point.note,
            }
            for point in points
        ]
    )


@tool
def get_forecast_accuracy() -> str:
    """Report how accurate the covers forecast has been on a rolling backtest.

    Use when the manager asks how much to trust the forecast. Returns the model's
    MAPE against a seasonal-naive baseline.
    """
    return json.dumps(backtest().to_dict())


@tool
def check_inventory(query: str = "") -> str:
    """Check current stock levels. Pass a product name or category to filter, or
    leave empty for everything.

    Use for "how much X do we have left". Returns quantity on hand, par level and
    days of cover per product.
    """
    session = SessionLocal()
    try:
        rows = session.query(Product, Stock).join(Stock, Stock.product_id == Product.id)
        needle = (query or "").strip().lower()
        if needle:
            rows = [
                (product, stock)
                for product, stock in rows.all()
                if needle in product.name.lower()
                or needle in product.category.lower()
                or needle in product.sku.lower()
            ]
        else:
            rows = rows.all()

        if not rows:
            return json.dumps({"matches": 0, "message": f"No product matching '{query}'."})

        return json.dumps({
            "matches": len(rows),
            "items": [
                {
                    "sku": product.sku,
                    "product": product.name,
                    "category": product.category,
                    "quantity_on_hand": round(float(stock.quantity_on_hand), 1),
                    "unit": product.unit,
                    "par_level": round(float(stock.par_level), 1),
                    "on_order": round(float(stock.on_order or 0), 1),
                    "below_par": float(stock.quantity_on_hand) < float(stock.par_level),
                    "supplier": product.supplier.name if product.supplier else None,
                    "lead_time_days": product.lead_time_days,
                }
                for product, stock in rows
            ],
        })
    finally:
        session.close()


@tool
def list_stockout_alerts(horizon_days: int = 14) -> str:
    """List products forecast to run out within the horizon, with the date to order by.

    Use for "what am I going to run short of" and before suggesting an order.
    Alerts are ranked by severity: critical means the order-by date has passed.
    """
    alerts = build_alerts(horizon_days=max(1, min(horizon_days, 60)))
    return json.dumps({
        "horizon_days": horizon_days,
        "alert_count": len(alerts),
        "alerts": [alert.to_dict() for alert in alerts],
    })


@tool
def find_supplier(product_or_category: str) -> str:
    """Find the registered supplier for a product or category.

    Use before creating a purchase order, to confirm who it goes to and what the
    order cut-off and delivery days are.
    """
    needle = product_or_category.strip().lower()
    session = SessionLocal()
    try:
        product = (
            session.query(Product)
            .filter(Product.name.ilike(f"%{needle}%") | Product.sku.ilike(f"%{needle}%"))
            .first()
        )
        if product and product.supplier:
            return json.dumps({
                "matched_on": "product",
                "product": product.name,
                "sku": product.sku,
                "supplier": _supplier_dict(product.supplier),
            })

        suppliers = [
            supplier
            for supplier in session.query(Supplier).filter(Supplier.active.is_(True)).all()
            if needle in (supplier.categories or "").lower()
        ]
        if suppliers:
            return json.dumps({
                "matched_on": "category",
                "suppliers": [_supplier_dict(supplier) for supplier in suppliers],
            })

        return json.dumps({
            "error": f"No registered supplier found for '{product_or_category}'.",
            "known_categories": sorted({
                category
                for supplier in session.query(Supplier).all()
                for category in (supplier.categories or "").split(",")
                if category
            }),
        })
    finally:
        session.close()


@tool
def create_purchase_order(items_json: str, requested_delivery_date: str = "") -> str:
    """Create a DRAFT purchase order and generate its PDF. Does not send anything.

    items_json is a JSON list like:
      [{"sku": "PRD-1001", "cases": 3}, {"sku": "PRD-2001", "cases": 2}]
    Quantities are in supplier cases. All items must come from the same supplier;
    call this once per supplier if the order spans several.

    Returns the PO number, the priced lines and the PDF path. Always show the
    manager the total and ask them to confirm before calling send_purchase_order.
    """
    try:
        requested = json.loads(items_json)
        if isinstance(requested, dict):
            requested = [requested]
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"items_json is not valid JSON: {exc}"})

    session = SessionLocal()
    try:
        lines: list[dict] = []
        suppliers: set[int] = set()

        for item in requested:
            sku = str(item.get("sku", "")).strip()
            product = session.query(Product).filter(Product.sku == sku).first()
            if not product:
                return json.dumps({"error": f"Unknown SKU '{sku}'. Call check_inventory first."})
            if not product.supplier:
                return json.dumps({"error": f"{product.name} has no registered supplier."})

            cases = int(item.get("cases") or item.get("quantity") or 0)
            if cases <= 0:
                return json.dumps({"error": f"Quantity for {sku} must be a positive number of cases."})
            if cases < (product.min_order_quantity or 1):
                cases = product.min_order_quantity

            suppliers.add(product.supplier.id)
            case_price = float(product.case_price_eur or 0.0)
            lines.append({
                "sku": product.sku,
                "product": product.name,
                "cases": cases,
                "quantity": cases * (product.units_per_case or 1),
                "unit": product.unit,
                "case_price_eur": case_price,
                "line_total_eur": round(cases * case_price, 2),
            })

        if len(suppliers) > 1:
            return json.dumps({
                "error": "Items span more than one supplier. Create one purchase "
                         "order per supplier.",
            })

        supplier = session.query(Supplier).get(suppliers.pop())
        total = round(sum(line["line_total_eur"] for line in lines), 2)
        po_number = f"PO-{dt.date.today():%Y%m%d}-{uuid.uuid4().hex[:5].upper()}"

        delivery_date = None
        if requested_delivery_date:
            try:
                delivery_date = dt.date.fromisoformat(requested_delivery_date)
            except ValueError:
                return json.dumps({"error": "requested_delivery_date must be YYYY-MM-DD."})

        pdf_path = build_po_pdf(
            po_number=po_number,
            supplier=_supplier_dict(supplier),
            lines=lines,
            total_eur=total,
            requested_delivery_date=delivery_date,
        )

        session.add(PurchaseOrder(
            po_number=po_number,
            supplier_id=supplier.id,
            requested_delivery_date=delivery_date,
            lines_json=json.dumps(lines),
            total_eur=total,
            status="draft",
            pdf_path=str(pdf_path),
        ))
        session.commit()

        return json.dumps({
            "po_number": po_number,
            "status": "draft",
            "supplier": supplier.name,
            "supplier_email": supplier.email,
            "order_cutoff_time": supplier.order_cutoff_time,
            "delivery_days": supplier.delivery_days,
            "lines": lines,
            "total_eur": total,
            "pdf_path": str(pdf_path),
            "next_step": "Show the manager the total and ask for confirmation, then "
                         "call send_purchase_order with this po_number.",
        })
    finally:
        session.close()


@tool
def send_purchase_order(po_number: str) -> str:
    """Email a DRAFT purchase order to its supplier with the PDF attached.

    Only call this after the manager has explicitly confirmed the order. This is
    the only tool that sends anything outside the building.
    """
    session = SessionLocal()
    try:
        order = (
            session.query(PurchaseOrder)
            .filter(PurchaseOrder.po_number == po_number.strip())
            .first()
        )
        if not order:
            return json.dumps({"error": f"No purchase order '{po_number}'."})
        if order.status == "sent":
            return json.dumps({
                "error": f"{po_number} was already sent to {order.sent_to} at "
                         f"{order.sent_at:%Y-%m-%d %H:%M}. Not sending twice.",
            })

        supplier = session.query(Supplier).get(order.supplier_id)
        lines = json.loads(order.lines_json)

        from pathlib import Path

        result = send_po_email(
            po_number=order.po_number,
            supplier=_supplier_dict(supplier),
            pdf_path=Path(order.pdf_path),
            lines=lines,
            total_eur=float(order.total_eur),
        )

        # A dry run is not a send. Marking it sent would let a later real send be
        # refused as a duplicate.
        if result["sent"]:
            order.status = "sent"
            order.sent_at = dt.datetime.now()
            order.sent_to = supplier.email

            for line in lines:
                stock = (
                    session.query(Stock)
                    .join(Product, Product.id == Stock.product_id)
                    .filter(Product.sku == line["sku"])
                    .first()
                )
                if stock:
                    stock.on_order = float(stock.on_order or 0.0) + float(line["quantity"])
            session.commit()

        return json.dumps({"po_number": order.po_number, **result})
    finally:
        session.close()


READ_TOOLS = [
    get_traffic_forecast,
    get_forecast_accuracy,
    check_inventory,
    list_stockout_alerts,
    find_supplier,
]
WRITE_TOOLS = [create_purchase_order, send_purchase_order]
ALL_TOOLS = READ_TOOLS + WRITE_TOOLS
