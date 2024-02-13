import pandas as pd
from datetime import datetime
from scripts.db.models import Order, sessionmaker
from scripts.trade.coingecko_coinbase_pairs import gecko_coinbase_currency_map

class Buyer:
    def __init__(self, context, data, model, prices):
        self.context = context
        self.data = data
        self.model = model
        self.prices = prices
    
    def buy(self, timestamp):
        predictions = self.model.predict(timestamp)
        filtered_predictions = self.filter_predictions(predictions, timestamp)

        if len(filtered_predictions) == 0:
            print("Nothing to buy...")
            return

        for idx, row in filtered_predictions.iterrows():
            self.create_order(row, timestamp)

    def filter_predictions(self, predictions, timestamp):
        filtered = predictions.copy()
        filtered = filtered[filtered['Mean Delta'] > self.context['max_delta']]
        filtered['Symbol'] = filtered['Coin'].map(lambda x: gecko_coinbase_currency_map.get(x, 'UNSUPPORTED'))
        filtered = filtered[filtered['Symbol'] != 'UNSUPPORTED']
        filtered = filtered[filtered.apply(lambda row: self.current_spread(row, timestamp) < self.context['max_spread'], axis=1)]

        return filtered


    def current_spread(self, prediction, timestamp):
        symbol = prediction['Symbol']
        current_ask = self.prices.ask_at_timestamp(symbol, timestamp)
        current_bid = self.prices.bid_at_timestamp(symbol, timestamp)
        return (current_ask - current_bid) / current_bid
        
    def select_buy(self, predictions):
         return predictions.sample(n=1).iloc[0]
    
    def create_order(self, selection, timestamp):
        symbol = gecko_coinbase_currency_map.get(selection['Coin'])
        product_id = f"{symbol}-USDC"

        order_id = f"{timestamp}_{symbol}"
        current_ask = self.prices.ask_at_timestamp(symbol, timestamp)
        current_bid = self.prices.bid_at_timestamp(symbol, timestamp)
        spread = round((current_ask - current_bid) / current_bid * 100, 3)
        quantity =  round(self.context['order_amount_usd'] / current_ask, 5)
        created_at = datetime.utcfromtimestamp(timestamp)

        Session = sessionmaker(bind=self.context['engine'])
        session = Session()

        print(f"Buying: {symbol} @ ${current_ask}, prediction: {selection['Max Delta']}%")
        order = Order(
            order_id=order_id,
            coinbase_product_id=product_id,
            quantity=quantity,
            purchase_price=current_ask,
            status="OPEN",
            action="SELL",
            stop_loss_percent=self.context['stop_loss_percent'],
            profit_percent=self.context['take_profit_percent'],
            predicted_max_delta=selection['Max Delta'],
            predicted_min_delta=selection['Min Delta'],
            purchase_time_spread_percent=spread,
            created_at=created_at
        )
        session.add(order)
        session.commit()
        session.close()