# python3 ./scripts/db/add_order.py BTC-USDC 1 10.0
import argparse
from datetime import datetime
from models import engine, Order, sessionmaker  # Import necessary components 

def add_order(coinbase_product_id, quantity, purchase_price): 
    Session = sessionmaker(bind=engine)
    session = Session()

    order = Order(
        coinbase_product_id=coinbase_product_id,
        quantity=quantity,
        purchase_price=purchase_price,
        status="OPEN",
        action="SELL",
        stop_loss_percent=0.955,
        profit_percent=1.1,
        created_at=datetime.now()  # Add the timestamp here
    )
    session.add(order)
    session.commit()
    print(f"Order added to database with ID: {order.id}")
    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add orders to the database')
    parser.add_argument('coinbase_product_id', help='Coinbase product ID')
    parser.add_argument('quantity', type=float, help='Order quantity')
    parser.add_argument('purchase_price', type=float, help='Purchase price for the order')
    args = parser.parse_args()

    add_order(args.coinbase_product_id, args.quantity, args.purchase_price)