from sqlalchemy import and_
from scripts.db.models import Order, sessionmaker
from datetime import datetime, timezone
import logging

class Seller():
    def __init__(self, context, broker, timesource, logger):
        self.broker = broker
        self.context = context
        self.timesource = timesource
        self.logger = logger
    
    def sell(self):
        orders = self.load_orders()
        if (len(orders) == 0):
            return
        
        prices = self.broker.prices()
        best_bids = {}
        for order in orders:
            symbol = order.coinbase_product_id.split("-")[0]
            price = prices.bid(symbol)
            best_bids[order.coinbase_product_id] = price

        for order in orders:
            best_bid = float(best_bids[order.coinbase_product_id])
            # print(f"Order: {order.coinbase_product_id}, {best_bid}")
            if self.should_trigger_order(order, best_bid):
                if self.context['sell_all_on_hit']:
                    self.sell_entire_holding(order, best_bid)
                else:
                    self.sell_order(order, best_bid)
            elif self.should_trigger_loss_recovery(order):
                self.set_recovery_mode(order, best_bid)
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

    def sell_order(self, order, best_bid):        
        self.logger.info(f"Selling: {order.coinbase_product_id}")
        success = self.broker.sell(order.id, order.coinbase_product_id, order.quantity, best_bid)
        if success:
            Session = sessionmaker(bind=self.context['engine'])
            session = Session()
            order_to_update = session.get(Order, order.id)
            if(best_bid > order.purchase_price):
                self.logger.info("Selling for a profit!!!")
            else:
                self.logger.info("Selling for a loss :(")

            timestamp_dt = datetime.fromtimestamp(self.timesource.now(), timezone.utc)

            order_to_update.status = "SOLD"
            order_to_update.sold_at = timestamp_dt
            session.commit()
            session.close()
        else:
            self.logger.error("Sell error, cancelling")

    def set_recovery_mode(self, order, best_bid):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()

        order_to_update = session.get(Order, order.id)
        new_price_target = best_bid * (self.context['raise_stoploss_threshold'])
        order_to_update.stop_loss_percent = (new_price_target - order_to_update.purchase_price) / order_to_update.purchase_price
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
    
    def should_trigger_loss_recovery(self, order):
        # Timezones, FML
        created_minutes_ago = (self.timesource.now() - (order.created_at.timestamp() - (7 * 60 * 60))) / 60
        if created_minutes_ago < 0:
            self.logger.warning("Negative time delta encountered.")
        if self.context['loss_recovery_after_minutes'] is not None and created_minutes_ago > self.context['loss_recovery_after_minutes'] and order.recovery_mode == False:
            return True
        return False

    def stop_loss_price(self, order):
        if order.stop_loss_percent > 1.0:
            return order.purchase_price * order.stop_loss_percent
        else:
            return order.purchase_price * (1 + order.stop_loss_percent)

    def profit_price(self, order):
        return order.purchase_price * order.profit_percent

    def load_orders(self):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        orders = session.query(Order).filter_by(status="OPEN").all()
        session.close()
        return orders
