import pandas as pd
from datetime import datetime
from scripts.db.models import engine, Order, sessionmaker
from scripts.trade.coingecko_coinbase_pairs import gecko_coinbase_currency_map

class Buyer:
    ORDER_AMOUNT_USD = 100.0
    STOP_LOSS_PERCENT = 0.96
    TAKE_PROFIT_PERCENT = 1.1
    MAX_DELTA = 2.0

    def __init__(self, data, model, prices):
        self.data = data
        self.model = model
        self.prices = prices
    
    def buy(self, timestamp):
        predictions = self.model.predict(timestamp)
        filtered_predictions = self.filter_predictions(predictions)

        if len(filtered_predictions) == 0:
            print("Nothing to buy...")
            return

        selection = self.select_buy(filtered_predictions)        
        self.create_order(selection, timestamp)

    def filter_predictions(self, predictions):
        filtered = predictions.copy()
        filtered = filtered[filtered['Max Delta'] > self.MAX_DELTA]
        filtered['Symbol'] = filtered['Coin'].map(lambda x: gecko_coinbase_currency_map.get(x, 'UNSUPPORTED'))
        filtered = filtered[filtered['Symbol'] != 'UNSUPPORTED']
        print(predictions)
        return filtered
    
    def select_buy(self, predictions):
         return predictions.sample(n=1).iloc[0]
    
    def create_order(self, selection, timestamp):
        symbol = gecko_coinbase_currency_map.get(selection['Coin'])
        product_id = f"{symbol}-USDC"

        order_id = f"{timestamp}_{symbol}"
        current_ask = self.prices.ask_at_timestamp(symbol, timestamp)
        current_bid = self.prices.bid_at_timestamp(symbol, timestamp)
        spread = round((current_ask - current_bid) / current_bid * 100, 3)
        quantity =  round(self.ORDER_AMOUNT_USD / current_ask, 5)
        created_at = datetime.utcfromtimestamp(timestamp)

        Session = sessionmaker(bind=engine)
        session = Session()

        print(f"Buying: {symbol} @ ${current_ask}, prediction: {selection['Max Delta']}%")
        order = Order(
            order_id=order_id,
            coinbase_product_id=product_id,
            quantity=quantity,
            purchase_price=current_ask,
            status="OPEN",
            action="SELL",
            stop_loss_percent=self.STOP_LOSS_PERCENT,
            profit_percent=self.TAKE_PROFIT_PERCENT,
            predicted_max_delta=selection['Max Delta'],
            predicted_min_delta=selection['Min Delta'],
            purchase_time_spread_percent=spread,
            created_at=created_at
        )
        session.add(order)
        session.commit()
        session.close()