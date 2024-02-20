import os
import sys
from datetime import datetime
from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
from scripts.trade.coingecko_coinbase_pairs import gecko_coinbase_currency_map
from scripts.live.prices import Prices


class Broker:
    def __init__(self, logger, context):
        self.logger = logger
        self.context = context

        api_key_name = os.getenv("COINBASE_API_KEY_NAME")
        private_key = os.getenv("COINBASE_PRIVATE_KEY")

        self.last_price_fetch_at = None
        self.prices_cached = None

        self.client = None
        try:
            self.client = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(
                api_key_name, private_key
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize the Coinbase Advanced Trade API Client: {e}"
            )
            sys.exit(1)

    def usdc_available(self):
        return self._holdings()["USDC"]

    def holding_available(self, symbol):
        holdings = self._holdings()
        if symbol in holdings:
            return holdings[symbol]
        return 0

    def buy(self, order_id, product_id, amount_usdc, highest_bid):
        product = self.client.get_product(product_id)
        base_decimal_precision = len(product.base_increment.split(".")[1])
        base_size = round(amount_usdc / highest_bid, base_decimal_precision)

        if self.context["live_trades"] == False:
            self.logger.info(
                f"SIMUALTED - Would have bought {amount_usdc} of {product_id} at {highest_bid} per unit"
            )
            return base_size

        quote_decimal_precision = len(product.quote_increment.split("."))
        bid = round(highest_bid, quote_decimal_precision)

        self.logger.info(f"Buying {base_size} of {product_id} at {bid} per unit")

        buy_order = self.client.create_limit_order(
            client_order_id=order_id,
            product_id=product_id,
            side=Side.BUY,
            base_size=base_size,
            limit_price=bid,
        )

        self.logger.info(f"Buy order status: {buy_order}")

        if buy_order.order_error is None:
            return base_size
        return 0

    def sell(self, order_id, product_id, amount_product, lowest_ask):
        if self.context["live_trades"] == False:
            self.logger.info(
                f"SIMUALTED - Would have sold {amount_product} of {product_id} at {lowest_ask} per unit"
            )
            return True

        product = self.client.get_product(product_id)

        client_order_id = f"{product_id}_#{order_id}"
        symbol = product_id.split("-")[0]
        base_size = min(self.holding_available(symbol), amount_product)

        decimal_precision = len(product.base_increment.split(".")[1])
        base_size = round(base_size, decimal_precision)
        sell_order = self.client.create_limit_order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=Side.SELL,
            base_size=base_size,
            limit_price=lowest_ask,
        )

        self.logger.info(f"Sell order status: {sell_order}")

        return sell_order.order_error is None

    def order_status(self, order_id):
        orders = list(
            filter(
                lambda o: o.client_order_id == order_id,
                self.client.list_orders_all().orders,
            )
        )
        if len(orders) == 0:
            return None
        return orders[0]

    def prices(self):
        if (
            self.prices_cached is not None
            and (datetime.now() - self.last_price_fetch_at).total_seconds() < 10
        ):
            return self.prices_cached

        self.logger.info("Fetching prices!")
        symbols = list(
            filter(lambda s: s != "UNSUPPORTED", gecko_coinbase_currency_map.values())
        )
        products = list(map(lambda s: "{}-USDC".format(s), symbols))

        asks, bids = self._fetch_current_prices(products)

        self.prices_cached = Prices(bids, asks)
        self.last_price_fetch_at = datetime.now()

        return self.prices_cached

    def _fetch_current_prices(self, product_ids):
        product_ids = filter(lambda p: p != "USDC-USDC", product_ids)
        prices = self.client.get_best_bid_ask(product_ids)

        bids = {"USDC-USDC": 1.0}
        asks = {"USDC-USDC": 1.0}
        for pricebook in prices.pricebooks:
            bids[pricebook.product_id] = float(pricebook.bids[0].price)
            asks[pricebook.product_id] = float(pricebook.asks[0].price)
        return asks, bids

    def _holdings(self):
        h = {}

        def process_account(account):
            balance = float(account.available_balance.value)
            if balance > 0:
                h[account.currency] = balance

        def process_response(response):
            for account in response.accounts:
                process_account(account)

        response = self.client.list_accounts()
        process_response(response)

        while response.has_next:
            response = self.client.list_accounts(cursor=response.cursor)
            process_response(response)

        return h
