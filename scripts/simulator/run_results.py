import sys
import numpy as np
from scripts.db.models import Order, sessionmaker, init_db_engine
import pandas as pd

class RunResults:
    def __init__(self, engine, best_bids):
        self.engine = engine
        self.best_bids = best_bids

    def get_results(self):
        FEES = 0.004

        with sessionmaker(bind=self.engine)() as session:
            orders = session.query(Order).all()

            sold_for_profit = [order for order in orders if order.status == "SOLD" and order.stop_loss_percent > 1.0]
            sold_at_loss = [order for order in orders if order.status == "SOLD" and order.stop_loss_percent <= 1.0]
            still_open = [order for order in orders if order.status == "OPEN"]

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
                    # Assuming data_bids is available in the scope of this class
                    latest_price = self.best_bids.iloc[-1][symbol]
                    sale_price = (order.quantity * latest_price) * (1 - FEES)

                    order_net = sale_price - purchase_price

                    total_purchases += purchase_price
                    total_net += order_net

            final_balance = total_purchases + total_net
            percent_change = round((final_balance - total_purchases) / total_purchases * 100, 2)

        return {
            'final_balance': final_balance,
            'total_purchases': total_purchases,
            'total_net': total_net,
            'percent_change': percent_change
        }