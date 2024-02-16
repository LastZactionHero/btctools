from sqlalchemy import and_
from scripts.db.models import Order, sessionmaker
from datetime import datetime

class Seller():
    def __init__(self, context, prices):
        self.prices = prices
        self.context = context
    
    def sell(self, timestamp):
        orders = self.load_orders()
        if (len(orders) == 0):
            return
        
        best_bids = {}
        for order in orders:
            symbol = order.coinbase_product_id.split("-")[0]
            price = self.prices.bid_at_timestamp(symbol, timestamp)
            best_bids[order.coinbase_product_id] = price

        for order in orders:
            best_bid = float(best_bids[order.coinbase_product_id])

            if self.should_trigger_order(order, best_bid):
                if self.context['sell_all_on_hit']:
                    self.sell_entire_holding(order, best_bid)
                else:
                    self.sell_order(order, best_bid, timestamp)
            if self.should_trigger_loss_recovery(order, timestamp):
                self.update_recovery_mode(order, best_bid)
            else:
                new_stoploss_value = self.adjust_stoploss(order, best_bid)
                if new_stoploss_value != order.stop_loss_percent:
                    self.update_order_stoploss(order, new_stoploss_value)

    def sell_entire_holding(self, order, best_bid):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        orders_to_sell = session.query(Order).filter(
            and_(Order.status == "OPEN", 
                Order.coinbase_product_id == order.coinbase_product_id))
        for order in orders_to_sell:
            order.status = "SOLD"
            order.stop_loss_percent = ((best_bid - order.purchase_price) / order.purchase_price) + 1
        session.commit()
        session.close()        

    def sell_order(self, order, best_bid, timestamp):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        order_to_update = session.get(Order, order.id)
        
        print(f"Selling: {order.coinbase_product_id}")
        if(best_bid > order.purchase_price):
            print("Selling for a profit!!!")
        else:
            print("Selling for a loss :(")

        timestamp_dt = datetime.utcfromtimestamp(timestamp)
        if timestamp_dt < order_to_update.created_at:
            import pdb; pdb.set_trace()

        order_to_update.status = "SOLD"
        order_to_update.sold_at = timestamp_dt
        session.commit()
        session.close()

    def update_recovery_mode(self, order, best_bid):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()

        order_to_update = session.get(Order, order.id)
        order_to_update.stop_loss_percent = best_bid * self.context['raise_stoploss_threshold']
        order_to_update.recovery_mode = True

        session.commit()
        session.close()
        

    def update_order_stoploss(self, order, new_stoploss_value):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()

        order_to_update = session.get(Order, order.id)
        order_to_update.stop_loss_percent = new_stoploss_value
        session.commit()
        session.close()
            
    def adjust_stoploss(self, order, best_bid):
        prev_value = order.stop_loss_percent
        next_value = order.stop_loss_percent

        if best_bid >= order.purchase_price * self.context['raise_stoploss_threshold']:
            next_value = ((best_bid - order.purchase_price) / order.purchase_price) + 1 - self.context['sell_stoploss_floor']

        return max(next_value, prev_value)

    def should_trigger_order(self, order, exchange_rate_usd):
        return exchange_rate_usd >= self.profit_price(order) or exchange_rate_usd <= self.stop_loss_price(order)
    
    def should_trigger_loss_recovery(self, order, timestamp):
        created_minutes_ago = (timestamp - order.created_at.timestamp()) / 60
        if self.context['loss_recovery_after_minutes'] is not None and self.context['loss_recovery_after_minutes'] > created_minutes_ago and order.recovery_mode == False:
            return True
        return False

    def stop_loss_price(self, order):
        return order.purchase_price * order.stop_loss_percent

    def profit_price(self, order):
        return order.purchase_price * order.profit_percent

    def load_orders(self):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        orders = session.query(Order).filter_by(status="OPEN").all()
        session.close()
        return orders
