import argparse
from models import engine, Order, sessionmaker

def delete_order(order_id):
    Session = sessionmaker(bind=engine)
    session = Session()

    # Updated line: Get the order using Session.get(): 
    order_to_delete = session.get(Order, order_id)  
  
    if order_to_delete:
        session.delete(order_to_delete)
        session.commit()
        print(f"Order with ID {order_id} deleted successfully.")
    else:
        print(f"Order with ID {order_id} not found.")

    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete an order from the database')
    parser.add_argument('order_id', type=int, help='ID of the order to delete')
    args = parser.parse_args()

    delete_order(args.order_id)
