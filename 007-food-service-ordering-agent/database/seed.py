"""Generate two years of covers history plus the product, stock and supplier tables.

The covers series is synthetic but not random noise. A staff restaurant has a very
specific signature and the forecasting model is only interesting if the data has
it: a strong weekday cycle, Friday down because people work from home, weekends
closed, a summer trough, and a slow upward trend as headcount grows.

Run with:  python -m database.seed
"""
from __future__ import annotations

import datetime as dt
import math
import random

from database.models import (
    Covers,
    Product,
    PurchaseOrder,
    SessionLocal,
    Stock,
    Supplier,
    init_db,
)

SITE_ID = "SITE-01"
HISTORY_DAYS = 730
RNG_SEED = 20260919

# Weekday multipliers. Monday is quiet, Tuesday to Thursday are the peak, Friday
# drops hard because of remote working.
WEEKDAY_FACTOR = {0: 0.92, 1: 1.06, 2: 1.08, 3: 1.02, 4: 0.72, 5: 0.0, 6: 0.0}

# Public holidays and the August shutdown. Closures have to be in the data, not
# smoothed out of it, because the model has to learn not to forecast covers on a
# day the site is shut.
FIXED_CLOSURES = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
AUGUST_SHUTDOWN = {(8, day) for day in range(1, 16)}

SUPPLIERS = [
    {
        "name": "Vallee Fresh Produce",
        "contact_name": "Claire Dumont",
        "email": "orders@vallee-fresh.example",
        "phone": "+33 4 72 00 11 22",
        "categories": "produce,salad",
        "order_cutoff_time": "15:00",
        "delivery_days": "Mon,Wed,Fri",
        "account_number": "VF-4471",
    },
    {
        "name": "Nord Beverages",
        "contact_name": "Marc Lefevre",
        "email": "commandes@nord-beverages.example",
        "phone": "+33 3 20 44 55 66",
        "categories": "beverage",
        "order_cutoff_time": "16:00",
        "delivery_days": "Tue,Thu",
        "account_number": "NB-9082",
    },
    {
        "name": "Atlas Dry Goods",
        "contact_name": "Sofia Benali",
        "email": "sales@atlas-drygoods.example",
        "phone": "+33 1 45 78 90 12",
        "categories": "dry_goods,bakery",
        "order_cutoff_time": "12:00",
        "delivery_days": "Mon,Thu",
        "account_number": "AD-2213",
    },
    {
        "name": "Cote Protein",
        "contact_name": "Julien Roche",
        "email": "orders@cote-protein.example",
        "phone": "+33 4 91 33 22 11",
        "categories": "protein,chilled",
        "order_cutoff_time": "14:00",
        "delivery_days": "Tue,Fri",
        "account_number": "CP-7756",
    },
]

# consumption_per_100_covers is what turns a covers forecast into a stock
# projection. Values are deliberately uneven so the alert list is not uniform.
# Prices are per CASE, matching units_per_case.
PRODUCTS = [
    # sku, name, category, unit, units/case, per-100-covers, lead, MOQ, shelf, case price, supplier
    ("PRD-1001", "Still water 50cl", "beverage", "case", 24, 11.0, 2, 2, 365, 8.40, "Nord Beverages"),
    ("PRD-1002", "Sparkling water 50cl", "beverage", "case", 24, 4.5, 2, 2, 365, 9.10, "Nord Beverages"),
    ("PRD-1003", "Orange juice 1L", "beverage", "case", 6, 2.2, 2, 1, 120, 11.70, "Nord Beverages"),
    ("PRD-2001", "Mixed salad leaves", "salad", "kg", 1, 6.8, 1, 5, 4, 5.20, "Vallee Fresh Produce"),
    ("PRD-2002", "Tomatoes", "produce", "kg", 1, 5.4, 1, 5, 7, 2.80, "Vallee Fresh Produce"),
    ("PRD-2003", "Potatoes", "produce", "kg", 25, 18.0, 2, 1, 30, 23.75, "Vallee Fresh Produce"),
    ("PRD-2004", "Carrots", "produce", "kg", 10, 4.1, 2, 1, 21, 13.00, "Vallee Fresh Produce"),
    ("PRD-3001", "Chicken breast", "protein", "kg", 5, 9.5, 2, 2, 5, 43.00, "Cote Protein"),
    ("PRD-3002", "Minced beef", "protein", "kg", 5, 6.2, 2, 2, 4, 52.00, "Cote Protein"),
    ("PRD-3003", "Salmon fillet", "protein", "kg", 4, 3.1, 3, 1, 3, 71.60, "Cote Protein"),
    ("PRD-3004", "Grated cheese", "chilled", "kg", 2, 2.7, 2, 2, 21, 14.60, "Cote Protein"),
    ("PRD-4001", "Baguette", "bakery", "unit", 20, 42.0, 1, 2, 1, 11.00, "Atlas Dry Goods"),
    ("PRD-4002", "Pasta penne", "dry_goods", "kg", 5, 7.4, 3, 2, 365, 9.25, "Atlas Dry Goods"),
    ("PRD-4003", "Rice long grain", "dry_goods", "kg", 10, 5.9, 3, 1, 365, 16.00, "Atlas Dry Goods"),
    ("PRD-4004", "Olive oil 5L", "dry_goods", "unit", 1, 0.6, 3, 1, 540, 28.50, "Atlas Dry Goods"),
    ("PRD-4005", "Paper napkins", "dry_goods", "case", 12, 3.8, 3, 1, 999, 14.20, "Atlas Dry Goods"),
]

# How many days of cover a site aims to hold, by category. This is what stops
# every product alerting at once: a 14-day horizon will exhaust anything held on
# a 3-day par, so without realistic par levels the alert list is all noise.
PAR_COVER_DAYS = {
    "dry_goods": 21,
    "beverage": 18,
    "bakery": 1,
    "produce": 4,
    "salad": 3,
    "protein": 5,
    "chilled": 7,
}

# Sites that will be short within the forecast horizon, so the demo has something
# to find. Everything else starts comfortably stocked.
TIGHT_STOCK = {"PRD-1001": 0.28, "PRD-2001": 0.22, "PRD-3001": 0.35, "PRD-4001": 0.30}


def is_closed(day: dt.date) -> tuple[bool, str | None]:
    if day.weekday() >= 5:
        return True, "weekend"
    if (day.month, day.day) in FIXED_CLOSURES:
        return True, "public holiday"
    if (day.month, day.day) in AUGUST_SHUTDOWN:
        return True, "summer shutdown"
    return False, None


def covers_for(day: dt.date, day_index: int, rng: random.Random) -> int:
    """Baseline covers with trend, annual seasonality, weekday shape and noise."""
    base = 480.0
    trend = day_index * 0.055                                  # headcount creeping up
    annual = -38.0 * math.cos(2 * math.pi * (day.timetuple().tm_yday - 15) / 365)
    weekday = WEEKDAY_FACTOR[day.weekday()]

    value = (base + trend + annual) * weekday
    value *= rng.gauss(1.0, 0.055)                              # ordinary day-to-day noise

    # Occasional one-off events: a site meeting, a training day, a strike.
    if rng.random() < 0.015:
        value *= rng.choice([0.55, 0.65, 1.28, 1.35])

    return max(0, int(round(value)))


def seed_covers(session, rng: random.Random) -> int:
    if session.query(Covers).count():
        print("Covers history already present, skipping.")
        return 0

    today = dt.date.today()
    start = today - dt.timedelta(days=HISTORY_DAYS)
    rows = []
    for offset in range(HISTORY_DAYS):
        day = start + dt.timedelta(days=offset)
        closed, note = is_closed(day)
        rows.append(
            Covers(
                site_id=SITE_ID,
                service_date=day,
                covers=0 if closed else covers_for(day, offset, rng),
                is_closed=closed,
                note=note,
            )
        )
    session.add_all(rows)
    return len(rows)


def seed_reference_data(session, rng: random.Random) -> tuple[int, int]:
    if session.query(Supplier).count():
        print("Suppliers and products already present, skipping.")
        return 0, 0

    suppliers = {}
    for payload in SUPPLIERS:
        supplier = Supplier(**payload)
        session.add(supplier)
        suppliers[payload["name"]] = supplier
    session.flush()

    # Average covers on an open day, used to size par levels sensibly.
    open_rows = [row.covers for row in session.query(Covers).filter(Covers.is_closed.is_(False))]
    avg_covers = sum(open_rows) / len(open_rows) if open_rows else 450

    for (sku, name, category, unit, per_case, rate, lead, moq,
         shelf, price, supplier_name) in PRODUCTS:
        product = Product(
            sku=sku,
            name=name,
            category=category,
            unit=unit,
            units_per_case=per_case,
            consumption_per_100_covers=rate,
            lead_time_days=lead,
            min_order_quantity=moq,
            shelf_life_days=shelf,
            case_price_eur=price,
            supplier=suppliers[supplier_name],
        )
        session.add(product)
        session.flush()

        daily_use = rate * avg_covers / 100.0
        # Par level is sized the way a site actually holds stock: dry goods and
        # drinks are bought in bulk and held for weeks, fresh produce is held for
        # days. Shelf life caps it, so a 1-day baguette never gets a two-week par.
        cover_days = min(PAR_COVER_DAYS.get(category, 5) + lead, max(shelf, 1))
        par = round(daily_use * cover_days, 1)

        fraction = TIGHT_STOCK.get(sku, rng.uniform(0.75, 1.15))
        session.add(
            Stock(
                product_id=product.id,
                site_id=SITE_ID,
                quantity_on_hand=round(par * fraction, 1),
                par_level=par,
                on_order=0.0,
            )
        )

    return len(SUPPLIERS), len(PRODUCTS)


def seed(reset_orders: bool = False) -> dict:
    init_db()
    rng = random.Random(RNG_SEED)
    session = SessionLocal()
    try:
        if reset_orders:
            session.query(PurchaseOrder).delete()

        covers_added = seed_covers(session, rng)
        session.flush()
        suppliers_added, products_added = seed_reference_data(session, rng)
        session.commit()

        total_covers = session.query(Covers).count()
        return {
            "covers_rows_added": covers_added,
            "covers_rows_total": total_covers,
            "suppliers_added": suppliers_added,
            "products_added": products_added,
        }
    finally:
        session.close()


if __name__ == "__main__":
    import json

    print("Seeding food service database ...")
    print(json.dumps(seed(), indent=2))
