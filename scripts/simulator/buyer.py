import pandas as pd
from datetime import datetime, timezone
from scripts.db.models import Order, sessionmaker
from sqlalchemy import and_
from scripts.trade.coingecko_coinbase_pairs import gecko_coinbase_currency_map

class Buyer:
    def __init__(self, context, data, model, broker, timesource):
        self.context = context
        self.data = data
        self.model = model
        self.broker = broker
        self.timesource = timesource
    
    def buy(self):
        predictions = self.model.predict()
        filtered_predictions = self.filter_predictions(predictions)

        if len(filtered_predictions) == 0:
            print("Nothing to buy...")
            return

        if self.context['single_buy'] == True:
            row = filtered_predictions.sort_values(by='Mean Delta', ascending=False).iloc[0]
            self.create_order(row)
        else:
            for idx, row in filtered_predictions.iterrows():
                self.create_order(row)

    def filter_predictions(self, predictions):
        filtered = predictions.copy()
        filtered = filtered[filtered['Mean Delta'] > self.context['max_delta']]
        filtered['Symbol'] = filtered['Coin'].map(lambda x: gecko_coinbase_currency_map.get(x, 'UNSUPPORTED'))
        filtered = filtered[filtered['Symbol'] != 'UNSUPPORTED']
        filtered = filtered[filtered.apply(lambda row: self.current_spread(row) < self.context['max_spread'], axis=1)]

        return filtered


    def current_spread(self, prediction):
        prices = self.broker.prices()

        symbol = prediction['Symbol']
        current_ask = prices.ask(symbol)
        current_bid = prices.bid(symbol)
        return (current_ask - current_bid) / current_bid
        
    def select_buy(self, predictions):
         return predictions.sample(n=1).iloc[0]
    
    def create_order(self, selection):
        prices = self.broker.prices()

        symbol = gecko_coinbase_currency_map.get(selection['Coin'])
        product_id = f"{symbol}-USDC"

        order_id = f"{self.timesource.now()}_{symbol}"
        current_ask = prices.ask(symbol)
        current_bid = prices.bid(symbol)
        spread = round((current_ask - current_bid) / current_bid * 100, 3)
        quantity =  round(self.context['order_amount_usd'] / current_ask, 5)
        created_at = datetime.fromtimestamp(self.timesource.now(), timezone.utc)

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

        