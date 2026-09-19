"""Operational database for one restaurant site.

Five tables, shaped the way the equivalent tables look in a real food-service
back office: covers history, products with a consumption rate, stock, suppliers,
and the purchase orders the agent creates.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'foodservice.db'}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Covers(Base):
    """Daily customer count. The series the forecast is built on."""

    __tablename__ = "covers"
    __table_args__ = (UniqueConstraint("site_id", "service_date", name="uq_site_date"),)

    id = Column(Integer, primary_key=True)
    site_id = Column(String, index=True, default="SITE-01")
    service_date = Column(Date, index=True)
    covers = Column(Integer)
    is_closed = Column(Boolean, default=False)
    note = Column(String, nullable=True)


class Product(Base):
    """A stocked item, with what it costs to run out of it.

    consumption_per_100_covers is the link between the traffic forecast and the
    stock projection. It is an empirical rate, not a recipe calculation.
    """

    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    sku = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    category = Column(String, index=True)
    unit = Column(String)                      # case, kg, tray, bottle
    units_per_case = Column(Integer, default=1)
    consumption_per_100_covers = Column(Float)
    lead_time_days = Column(Integer, default=2)
    min_order_quantity = Column(Integer, default=1)
    shelf_life_days = Column(Integer, default=30)
    # Price of one purchasing case, not one unit. Suppliers invoice by the
    # case, so mixing the two produces order totals that are wrong by a factor
    # of units_per_case.
    case_price_eur = Column(Float)
    preferred_supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=True)

    supplier = relationship("Supplier", back_populates="products")
    stock = relationship("Stock", back_populates="product", uselist=False)


class Stock(Base):
    __tablename__ = "stock"

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), unique=True)
    site_id = Column(String, index=True, default="SITE-01")
    quantity_on_hand = Column(Float)
    par_level = Column(Float)                  # the level the site aims to hold
    on_order = Column(Float, default=0.0)
    last_counted_at = Column(DateTime, default=dt.datetime.now)

    product = relationship("Product", back_populates="stock")


class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    contact_name = Column(String)
    email = Column(String)
    phone = Column(String)
    categories = Column(String)                # comma-separated
    order_cutoff_time = Column(String, default="16:00")
    delivery_days = Column(String, default="Mon,Tue,Wed,Thu,Fri")
    account_number = Column(String)
    active = Column(Boolean, default=True)

    products = relationship("Product", back_populates="supplier")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True)
    po_number = Column(String, unique=True, index=True)
    site_id = Column(String, default="SITE-01")
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    created_at = Column(DateTime, default=dt.datetime.now)
    requested_delivery_date = Column(Date, nullable=True)
    lines_json = Column(Text)                  # JSON list of order lines
    total_eur = Column(Float)
    status = Column(String, default="draft")   # draft, sent, failed
    pdf_path = Column(String, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    sent_to = Column(String, nullable=True)
    created_by = Column(String, default="agent")

    supplier = relationship("Supplier")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
