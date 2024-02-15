from scripts.db.models import Order, sessionmaker, init_db_engine
from datetime import datetime
import numpy as np
import sys
import pandas as pd

FEES = 0.004
FILENAME_BIDS = "./data/bids.csv"
db_filename = sys.argv[1]
engine = init_db_engine(db_filename)

Session = sessionmaker(bind=engine)
session = Session()
orders = session.query(Order).all()

sold_hold_times = []
open_hold_times = []
profit_hold_times = []
loss_hold_times = []

data_bids = pd.read_csv(FILENAME_BIDS)

last_timestamp = datetime.utcfromtimestamp(data_bids['timestamp'].values[-1])

for order in orders:
    hold_time = (order.sold_at - order.created_at).total_seconds() if order.status == "SOLD" else (last_timestamp - order.created_at).total_seconds()
    hold_time = hold_time / (60 * 60 * 24)  # Converting seconds to days
    if order.status == "SOLD":
        sold_hold_times.append(hold_time)
        purchase_price = ((order.purchase_price * order.quantity) * (1 + FEES))
        sale_price = (order.purchase_price * order.quantity * order.stop_loss_percent) * (1 - FEES)
        order_net = sale_price - purchase_price
        if order_net > 0:
            profit_hold_times.append(hold_time)
        elif order_net < 0:
            loss_hold_times.append(hold_time)
    elif order.status == "OPEN":
        open_hold_times.append(hold_time)

session.close()

def print_stats(hold_times, label):
    print(f"\nStats for {label} orders:")
    print(f"Avg hold time: {np.average(hold_times):.2f} days")
    print(f"Median hold time: {np.median(hold_times):.2f} days")
    print(f"Max hold time: {np.max(hold_times):.2f} days")
    print(f"Min hold time: {np.min(hold_times):.2f} days")
    print(f"Std Dev hold time: {np.std(hold_times):.2f} days\n")

print_stats(sold_hold_times, "SOLD")
print_stats(open_hold_times, "OPEN")
print_stats(profit_hold_times, "Profit")
print_stats(loss_hold_times, "Loss")
