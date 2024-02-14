from sqlalchemy import and_
from scripts.db.models import Order, sessionmaker

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
                    self.sell_order(order, best_bid)
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
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        order_to_update = session.get(Order, order.id)
        
        print(f"Selling: {order.coinbase_product_id}")
        if(best_bid > order.purchase_price):
            print("Selling for a profit!!!")
        else:
            print("Selling for a loss :(")

        order_to_update.status = "SOLD"
        session.commit()
        session.close()

    def update_order_stoploss(self, order, new_stoploss_value):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()

        order_to_update = session.get(Order, order.id)  # Get the Order from the database
        order_to_update.stop_loss_percent = new_stoploss_value   # Update the stop-loss
        session.commit()    # Commit the change to the database
        session.close()
            
    def adjust_stoploss(self, order, best_bid):
        prev_value = order.stop_loss_percent
        next_value = order.stop_loss_percent

        if best_bid >= order.purchase_price * self.context['raise_stoploss_threshold']:
            next_value = ((best_bid - order.purchase_price) / order.purchase_price) + 1 - self.context['sell_stoploss_floor']

        return max(next_value, prev_value)

    def should_trigger_order(self, order, exchange_rate_usd):
        return exchange_rate_usd >= self.profit_price(order) or exchange_rate_usd <= self.stop_loss_price(order)
    
    def stop_loss_price(self, order):
        return order.purchase_price * order.stop_loss_percent

    def profit_price(self, order):
        return order.purchase_price * order.profit_percent

    def load_orders(self):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        orders = session.query(Order).filter_by(status="OPEN").all()  # Filter only OPEN orders
        session.close()
        return orders
