from database.models import engine, Base, SessionLocal, Order
from faker import Faker
import random
from datetime import timedelta


Base.metadata.create_all(bind=engine)

fake = Faker()
db = SessionLocal()

STATUSES = ["Processing", "Shipped", "Delivered", "Cancelled", "Delayed"]

def seed_data(num_records=50):
    if db.query(Order).count() > 0:
        print("Database already has data. Skipping...")
        return

    print("Generating mock orders...")
    for _ in range(num_records):
        created = fake.date_time_between(start_date='-30d', end_date='now')
        delivery = created + timedelta(days=random.randint(2, 10))
        
        order = Order(
            order_number=f"ORD-{fake.random_int(min=10000, max=99999)}",
            customer_name=fake.name(),
            status=random.choice(STATUSES),
            shipping_address=fake.address().replace('\n', ', '),
            created_at=created,
            estimated_delivery=delivery
        )
        db.add(order)
    
    db.commit()
    print(f"Successfully added {num_records} orders to the database!")
    

    sample_order = db.query(Order).first()
    print(f"\n--- SAMPLE ORDER FOR TESTING ---")
    print(f"Order Number: {sample_order.order_number}")
    print(f"Customer: {sample_order.customer_name}")
    print(f"Status: {sample_order.status}")

if __name__ == "__main__":
    seed_data()