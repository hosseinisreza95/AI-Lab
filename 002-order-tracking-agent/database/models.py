from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime


DATABASE_URL = "sqlite:///./orders.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String, unique=True, index=True) # مثلا ORD-12345
    customer_name = Column(String)
    status = Column(String) # Processing, Shipped, Delivered, Delayed, Cancelled
    shipping_address = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    estimated_delivery = Column(DateTime, nullable=True)