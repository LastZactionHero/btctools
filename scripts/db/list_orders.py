from models import Order, sessionmaker, init_db_engine
from prettytable import PrettyTable  # Import the PrettyTable library
import sys

engine = init_db_engine(sys.argv[1])

def list_orders(status):
    Session = sessionmaker(bind=engine)
    session = Session()

    orders = None
    if status == 'OPEN' or status == 'SOLD':
        orders = session.query(Order).filter(Order.status == status)
    else:
        orders = session.query(Order).all()

    table = PrettyTable()  # Create a PrettyTable instance 
    table.field_names = ["ID", "Status", "Action", "Coinbase Product ID", 
                         "Quantity", "Price", "Spread", "Stop Loss", "Target", 
                         "Max D", "Min D",
                         "Created At", "Sold At"]

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
                    #    round(order.num_predictions_over_hit, 2),
                    #    round(order.max_delta_average, 2),
                    #    round(order.max_delta_std, 2),
                    #    round(order.min_delta_average, 2),
                    #    round(order.min_delta_std, 2),
                       order.created_at,
                       order.sold_at])

    session.close()
    print(table)  # Print the formatted table

if __name__ == "__main__":
    
    status = sys.argv[2] if len(sys.argv) > 2  else None
    list_orders(status)


