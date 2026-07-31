from langchain_core.tools import tool
from database.models import SessionLocal, Order

@tool
def get_order_status(order_number: str) -> str:
    """
    Use this tool when a user asks for the status of their order.
    The input should be a valid order number (e.g., ORD-12345).
    This tool queries the database and returns the order details.
    """
    db = SessionLocal()
    try:
        # Search for the order in the database by order number
        order = db.query(Order).filter(Order.order_number == order_number).first()
        
        if order:
            # If the order is found, return the details to the language model
            estimated_date = order.estimated_delivery.strftime('%Y-%m-%d') if order.estimated_delivery else 'Unknown'
            return (f"Order {order.order_number} for {order.customer_name} is currently in '{order.status}' status. "
                    f"Shipping address: {order.shipping_address}. "
                    f"Estimated delivery date: {estimated_date}.")
        else:
            return f"Sorry, no order found with the number '{order_number}' in the system. Please verify the order number."
    except Exception as e:
        return f"Database error occurred: {str(e)}"
    finally:
        db.close()