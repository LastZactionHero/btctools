import sys
import numpy as np
from scripts.db.models import Order, sessionmaker, init_db_engine

FEES = 0.004

db_filename = sys.argv[1]
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

print(f"Orders at profit: {len(sold_for_profit)}")
print(f"Orders at loss:   {len(sold_at_loss)}")
print(f"Still open:       {len(still_open)}")

total_purchases = 0
total_net = 0

for order in orders:
    if order.status != 'SOLD':
        continue

    purchase_price = ((order.purchase_price * order.quantity) * (1 + FEES))
    # print(f"purchased: {purchase_price}")
    sale_price = (order.purchase_price * order.quantity * order.stop_loss_percent) * (1 - FEES)
    # print(f"sold:      {sale_price}")
    order_net = sale_price - purchase_price
    # print(f"net:       {order_net}")

    # print(f"Net: ${round(order_net, 2)}, Purchase: ${round(purchase_price, 2)}, Sale: ${round(sale_price, 2)}")

    total_purchases += purchase_price
    total_net += order_net

final_balance = total_purchases + total_net                
print(total_purchases)
print(total_net)

percent_change = round((final_balance - total_purchases) / total_purchases * 100, 2)
print(percent_change)

avg_profit_spread = round(np.average(profit_spread), 2)
avg_loss_spread = round(np.average(loss_spread), 2)
median_profit_spread = round(np.median(profit_spread), 2)
median_loss_spread = round(np.median(loss_spread), 2)
std_profit_spread = round(np.std(profit_spread), 2)
std_loss_spread = round(np.std(loss_spread), 2)

median_profit_max_delta = round(np.median(profit_est_delta), 2)
median_loss_max_delta = round(np.median(loss_est_delta), 2)
std_profit_max_delta = round(np.std(profit_est_delta), 2)
std_loss_max_delta = round(np.std(loss_est_delta), 2)

print(f"Avg Profit Spread: {avg_profit_spread}")
print(f"Avg Loss Spread:   {avg_loss_spread}")
print(f"Med Profit Spread: {median_profit_spread}")
print(f"Med Loss Spread:   {median_loss_spread}")
print(f"Std Profit Spread: {std_profit_spread}")
print(f"Std Loss Spread:   {std_loss_spread}")
print(f"Med Profit Max D:  {median_profit_max_delta}")
print(f"Med Loss Max D:    {median_loss_max_delta}")
print(f"Std Profit Max D:  {std_profit_max_delta}")
print(f"Std Loss Max D:    {std_loss_max_delta}")

# print("Profit Spreads")
# print(np.sort(profit_spread))

# print("Loss Spreads")
# print(np.sort(loss_spread))

for order in orders:
    if order.status != 'SOLD':
        continue
    if order.purchase_time_spread_percent > 0.01:
        continue
    # if order.predicted_max_delta < 3.76:
    #     continue
    print(order.stop_loss_percent)