import glob
import re
import sys
import pandas as pd
from scripts.db.models import Order, sessionmaker, init_db_engine

experiment_name = sys.argv[1]

FEES = 0.004
RUN_DURATION_MINUTES = 5 * 60 * 24
FILENAME_BIDS = "./data/bids.csv"
data_bids = pd.read_csv(FILENAME_BIDS)

# Create a regex pattern for floating point values
pattern = re.compile(r'(\d+)_(\d+)\.db$')

# Use glob to get all db files for the experiment
files = glob.glob(f'./db/exp_{experiment_name}_*.db')

# Filter files based on the regex pattern
matching_files = [f for f in files if pattern.search(f)]

def parse_float_from_filename(filename):
    match = pattern.search(filename)
    if match:
        # Convert the matched groups to a floating point value
        return float(f"{match.group(1)}.{match.group(2)}")
    return None



data = []
columns = [experiment_name, 'count_sold_for_profit', 'count_sold_at_loss', 'count_still_open', 'total_purchases', 'total_net', 'final_balance', 'percent_change']

for db_filename in matching_files:
    parameter_value = parse_float_from_filename(db_filename)
    engine = init_db_engine(db_filename)
    Session = sessionmaker(bind=engine)
    session = Session()
    orders = session.query(Order).all()

    sold_for_profit = []
    sold_at_loss = []
    still_open = []
    profit_spread = []
    loss_spread = []
    profit_est_delta = []
    loss_est_delta = []

    for order in orders:
        if order.status == "SOLD":
            if order.stop_loss_percent > 1.0:
                sold_for_profit.append(order)
                profit_spread.append(order.purchase_time_spread_percent)
                profit_est_delta.append(order.predicted_max_delta)
            else:
                sold_at_loss.append(order)
                loss_spread.append(order.purchase_time_spread_percent)
                loss_est_delta.append(order.predicted_max_delta)
        elif order.status == "OPEN":
            still_open.append(order)

    session.close()

    total_purchases = 0
    total_net = 0

    for order in orders:
        if order.status == 'SOLD':
            purchase_price = ((order.purchase_price * order.quantity) * (1 + FEES))
            sale_price = (order.purchase_price * order.quantity * order.stop_loss_percent) * (1 - FEES)
            order_net = sale_price - purchase_price

            total_purchases += purchase_price
            total_net += order_net
        else:
            # What would be the price if it sold now?
            purchase_price = ((order.purchase_price * order.quantity) * (1 + FEES))

            symbol = order.coinbase_product_id.split("-")[0]

            latest_price = data_bids.iloc[-1][symbol]
            if RUN_DURATION_MINUTES is not None:
                start_timestamp = sorted(orders, key=lambda o: o.created_at)[0].created_at.timestamp()
                end_timestamp = start_timestamp + RUN_DURATION_MINUTES * 60
                data_bids[data_bids['timestamp'] > end_timestamp]
                latest_price = data_bids[data_bids['timestamp'] > end_timestamp].iloc[0][symbol]
            
            sale_price = (order.quantity * latest_price) * (1 - FEES)

            order_net = sale_price - purchase_price

            total_purchases += purchase_price
            total_net += order_net

    final_balance = total_purchases + total_net
    percent_change = round((final_balance - total_purchases) / total_purchases * 100, 2)

    data.append([parameter_value, len(sold_for_profit), len(sold_at_loss), len(still_open), total_purchases, total_net, final_balance, percent_change])

df = pd.DataFrame(data, columns=columns)
df = df.sort_values(by=[experiment_name], ascending=True)
print(df)