from models import engine, Order, sessionmaker
from prettytable import PrettyTable  # Import the PrettyTable library

def list_orders():
    Session = sessionmaker(bind=engine)
    session = Session()

    orders = session.query(Order).all()
    table = PrettyTable()  # Create a PrettyTable instance 
    table.field_names = ["ID", "Status", "Action", "Coinbase Product ID", 
                         "Quantity", "Price", "Stop Loss", "Profit Target", "Created At"]

    for order in orders:
        table.add_row([order.id, order.status, order.action, 
                       order.coinbase_product_id, order.quantity,
                       order.purchase_price, order.stop_loss_percent, 
                       order.profit_percent, order.created_at]n)

    session.close()
    print(table)  # Print the formatted table

if __name__ == "__main__":
    list_orders()
    list_holdings()
