import pandas as pd
from datetime import datetime, timezone
from scripts.db.models import Order, sessionmaker
from sqlalchemy import and_
from scripts.trade.coingecko_coinbase_pairs import gecko_coinbase_currency_map

class Buyer:
    def __init__(self, context, model, broker, timesource, logger):
        self.context = context
        self.model = model
        self.broker = broker
        self.timesource = timesource
        self.logger = logger
    
    def buy(self, price_data, latest):
        predictions = self.model.predict(price_data, latest)
        self.logger.info("Predictions:")
        self.logger.info(predictions)
        filtered_predictions = self.filter_predictions(price_data, predictions)

        if len(filtered_predictions) == 0:
            self.logger.info("Nothing to buy...")
            return

        if self.context['single_buy'] == True:
            row = filtered_predictions.sort_values(by='Mean Delta', ascending=False).iloc[0]
            self.create_order(row)
        else:
            for idx, row in filtered_predictions.iterrows():
                self.create_order(row)

    def min_time_above_threshold(self, price_data, row):
        latest_price = row['Latest']
        coin = row['Coin']

        prior_prices = price_data[coin][0-self.context['time_above_minutes_to_review']:-1][price_data[coin] > latest_price]
        time_above =  100 * float(len(prior_prices)) / self.context['time_above_minutes_to_review']
        return time_above > self.context['time_above_threshold']

    def filter_predictions(self, price_data, predictions):
        filtered = predictions.copy()
        
        filtered['Symbol'] = filtered['Coin'].map(lambda x: gecko_coinbase_currency_map.get(x, 'UNSUPPORTED'))
        filtered = self.filter_restricted_coins(filtered)
        filtered = filtered[filtered['Mean Delta'] > self.context['max_delta']]
        filtered = filtered[filtered['Symbol'] != 'UNSUPPORTED']
        filtered = filtered[filtered.apply(lambda row: self.current_spread(row) < self.context['max_spread'], axis=1)]
        filtered = filtered[filtered.apply(lambda row: self.min_time_above_threshold(price_data, row), axis=1)]
        filtered = self.filter_repeated_orders(filtered)
        
        
        return filtered
    
    def filter_repeated_orders(self, filtered):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        repeat_order_count = self.context['max_repeat_orders']
        last_orders = session.query(Order).order_by(Order.id.desc()).limit(repeat_order_count).all()
        last_orders_symbols = list(map(lambda o: o.coinbase_product_id.split('-')[0], last_orders))
        
        if len(last_orders_symbols) < repeat_order_count:
            session.commit()
            session.close()
            return filtered
        
        repeat_order = all(x == last_orders_symbols[0] for x in last_orders_symbols)

        if repeat_order:
            filtered = filtered[filtered['Symbol'] != last_orders_symbols[0]]
            self.logger.info(f"Filtering out repeat order {last_orders_symbols[0]}")
        session.commit()
        session.close()
        
        return filtered

    def filter_restricted_coins(self, filtered):
        filtered = filtered[~filtered['Symbol'].isin(self.context['restricted_coins'])]
        return filtered

    def current_spread(self, prediction):
        prices = self.broker.prices()

        symbol = prediction['Symbol']
        current_ask = prices.ask(symbol)
        current_bid = prices.bid(symbol)
        spread = (current_ask - current_bid) / current_bid
        return spread
        
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

        bid = (current_ask + current_bid) / 2

        self.logger.info(f"Buying: {symbol} @ ${bid}, prediction: {selection['Max Delta']}%")
        
        base_size = self.broker.buy(order_id, product_id, self.context['order_amount_usd'], bid)

        if base_size > 0:
            Session = sessionmaker(bind=self.context['engine'])
            session = Session()
            order = Order(
                order_id=order_id,
                coinbase_product_id=product_id,
                quantity=base_size,
                purchase_price=bid,
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
            self.logger.info("Order created successfully.")
        else:
            self.logger.error("Buy error, cancelling")
