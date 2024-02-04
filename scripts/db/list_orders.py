from models import engine, Order, sessionmaker
from prettytable import PrettyTable  # Import the PrettyTable library

def list_orders():
    Session = sessionmaker(bind=engine)
    session = Session()

    # orders = session.query(Order).all()
    orders = session.query(Order).filter_by(status="OPEN")
    table = PrettyTable()  # Create a PrettyTable instance 
    table.field_names = ["ID", "Status", "Action", "Coinbase Product ID", 
                         "Quantity", "Price", "Spread", "Stop Loss", "Target", 
                         "Max D", "Min D",
                         "# Pred > Hit", "Max D Avg",
                         "Max D STD", "Min D Avg", "Min D STD",
                         "Created At"]

    for order in orders:
        table.add_row([order.id, 
                       order.status, 
                       order.action, 
                       order.coinbase_product_id,
                       round(order.quantity, 2),
                       order.purchase_price, 
                       order.purchase_time_spread_percent,
                       order.stop_loss_percent, 
                       round(order.profit_percent, 2), 
                       round(order.predicted_max_delta, 2),
                       round(order.predicted_min_delta, 2),
                       round(order.num_predictions_over_hit, 2),
                       round(order.max_delta_average, 2),
                       round(order.max_delta_std, 2),
                       round(order.min_delta_average, 2),
                       round(order.min_delta_std, 2),
                       order.created_at])

    session.close()
    print(table)  # Print the formatted table

if __name__ == "__main__":
    list_orders()


