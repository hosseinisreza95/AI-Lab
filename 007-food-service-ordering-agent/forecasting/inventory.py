"""The inventory engine: forecast covers, project stock, raise alerts.

This is the join that makes the forecast operationally useful. On its own,
"we expect 512 covers on Thursday" is trivia. Multiplied by a per-product
consumption rate and subtracted from what is on the shelf, it becomes "you will
run out of salad leaves on Thursday, and the lead time is one day, so order
today".

The alert is deliberately expressed against the order-by date rather than the
stock-out date. A manager told they will run out next Thursday and not told that
the supplier needs three days will still run out.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass

from database.models import Product, SessionLocal, Stock
from forecasting.model import ForecastPoint, forecast_covers

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2}


@dataclass
class DayProjection:
    service_date: dt.date
    predicted_covers: int
    consumption: float
    closing_stock: float


@dataclass
class StockAlert:
    sku: str
    product: str
    category: str
    unit: str
    quantity_on_hand: float
    par_level: float
    daily_consumption_estimate: float
    stockout_date: dt.date | None
    days_of_cover: float
    lead_time_days: int
    order_by_date: dt.date | None
    suggested_order_quantity: float
    suggested_order_cases: int
    estimated_cost_eur: float
    supplier: str | None
    supplier_email: str | None
    severity: str
    reason: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("stockout_date", "order_by_date"):
            if payload[key] is not None:
                payload[key] = payload[key].isoformat()
        return payload


def project_product(
    product: Product, stock: Stock, forecast: list[ForecastPoint]
) -> list[DayProjection]:
    """Walk the forecast forward, draining stock a day at a time."""
    on_hand = float(stock.quantity_on_hand or 0.0) + float(stock.on_order or 0.0)
    projections: list[DayProjection] = []

    for point in forecast:
        consumption = (
            0.0
            if point.is_closed
            else point.predicted_covers * product.consumption_per_100_covers / 100.0
        )
        on_hand -= consumption
        projections.append(
            DayProjection(
                service_date=point.service_date,
                predicted_covers=point.predicted_covers,
                consumption=round(consumption, 2),
                closing_stock=round(on_hand, 2),
            )
        )
    return projections


def _severity(days_until_order_by: float | None, days_of_cover: float) -> str:
    if days_until_order_by is not None and days_until_order_by <= 0:
        return "critical"
    if days_of_cover <= 3:
        return "high"
    return "medium"


def build_alerts(
    horizon_days: int = 14,
    site_id: str = "SITE-01",
    include_all: bool = False,
    action_window_days: int = 7,
) -> list[StockAlert]:
    """One pass over every product. Returns only what needs ordering soon.

    The filter is the order-by date, not the stock-out date. Over a long enough
    horizon every product runs out, so "will run out within 14 days" flags the
    whole catalogue and tells the manager nothing. "You must place this order
    within the next 7 days or you will run out" is a worklist.
    """
    forecast = forecast_covers(horizon_days=horizon_days, site_id=site_id)
    today = dt.date.today()

    session = SessionLocal()
    try:
        rows = (
            session.query(Product, Stock)
            .join(Stock, Stock.product_id == Product.id)
            .filter(Stock.site_id == site_id)
            .all()
        )

        alerts: list[StockAlert] = []
        for product, stock in rows:
            projections = project_product(product, stock, forecast)

            stockout = next((p.service_date for p in projections if p.closing_stock <= 0), None)
            open_days = [p for p in projections if p.consumption > 0]
            daily_estimate = (
                sum(p.consumption for p in open_days) / len(open_days) if open_days else 0.0
            )

            if stockout is None:
                if not include_all:
                    continue
                days_of_cover = float(horizon_days)
                order_by = None
                days_until_order_by = None
            else:
                days_of_cover = (stockout - today).days
                order_by = stockout - dt.timedelta(days=product.lead_time_days)
                days_until_order_by = (order_by - today).days
                if not include_all and days_until_order_by > action_window_days:
                    # It will run out inside the horizon, but not before the next
                    # ordinary ordering round. Not the manager's problem today.
                    continue

            # Order back up to par, plus what the horizon will consume beyond it,
            # then round up to whole cases because that is how suppliers sell.
            deficit = max(
                float(stock.par_level) - float(stock.quantity_on_hand),
                sum(p.consumption for p in projections) - float(stock.quantity_on_hand),
            )
            # Never order past the shelf life; a two-week order of 1-day bread is
            # a worse outcome than the stock-out it prevents.
            shelf_cap = daily_estimate * max(product.shelf_life_days, 1)
            quantity = max(0.0, min(deficit, shelf_cap))

            units_per_case = max(product.units_per_case or 1, 1)
            cases = max(math.ceil(quantity / units_per_case), product.min_order_quantity or 1)
            ordered_units = cases * units_per_case

            supplier = product.supplier
            alerts.append(
                StockAlert(
                    sku=product.sku,
                    product=product.name,
                    category=product.category,
                    unit=product.unit,
                    quantity_on_hand=round(float(stock.quantity_on_hand), 1),
                    par_level=round(float(stock.par_level), 1),
                    daily_consumption_estimate=round(daily_estimate, 2),
                    stockout_date=stockout,
                    days_of_cover=round(days_of_cover, 1),
                    lead_time_days=product.lead_time_days,
                    order_by_date=order_by,
                    suggested_order_quantity=round(ordered_units, 1),
                    suggested_order_cases=cases,
                    estimated_cost_eur=round(cases * float(product.case_price_eur or 0.0), 2),
                    supplier=supplier.name if supplier else None,
                    supplier_email=supplier.email if supplier else None,
                    severity=_severity(days_until_order_by, days_of_cover),
                    reason=(
                        f"Projected to run out on {stockout.isoformat()} at "
                        f"{daily_estimate:.1f} {product.unit}/day against forecast covers."
                        if stockout
                        else "Within cover for the full horizon."
                    ),
                )
            )
    finally:
        session.close()

    alerts.sort(
        key=lambda alert: (SEVERITY_ORDER.get(alert.severity, 9), alert.days_of_cover)
    )
    return alerts


if __name__ == "__main__":
    import json

    found = build_alerts()
    print(f"{len(found)} product(s) must be ordered within the next 7 days:\n")
    print(json.dumps([alert.to_dict() for alert in found], indent=2))
