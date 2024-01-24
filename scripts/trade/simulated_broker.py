# fmt: off
import sys
sys.path.append("./scripts")
from datetime import datetime
from db.models import engine, SimulatedHolding, sessionmaker

TAKER_FEE = 0.004

class Holding:
    def __init__(self, coinbase_product_id, quantity):
        self.product_id = coinbase_product_id
        self.currency = coinbase_product_id.split('-')[0]
        self.balance_coin = quantity

        self.balance_usd = None
        self.allocation = None

        if coinbase_product_id == "USDC-USDC":
            self.balance_usd = quantity
            self.allocation = 0.0
        
    def set_best_bid(self, bid):
        self.balance_usd = self.balance_coin * bid

    def set_allocation(self, allocation):
        self.allocation = allocation

    def set_balance_usdc(self, usdc):
        self.balance_coin = usdc
        self.balance_usd = usdc


class SimulatedBroker:
    def __init__(self, coinbase_client):
        self.coinbase_client = coinbase_client

    def buy(self, client_order_id, product_id, limit_price, amount_usd):
        Session = sessionmaker(bind=engine)
        session = Session()

        base_size = amount_usd / limit_price

        holding = SimulatedHolding(
            coinbase_product_id=product_id,
            order_id=client_order_id,
            quantity=base_size,
            purchase_price=limit_price,
            created_at=datetime.now()
        )

        if product_id != "USDC-USDC":
            holding_usdc = session.query(SimulatedHolding).filter(
                SimulatedHolding.coinbase_product_id == "USDC-USDC")[0]
            holding_usdc.quantity = holding_usdc.quantity - amount_usd * (1 + TAKER_FEE)

        session.add(holding)
        session.commit()
        session.close()

        return { 'quantity': base_size, 'purchase_price': limit_price, 'order_id': client_order_id }

    def sell(self, client_order_id, buy_order_id, product_id, limit_price, base_size):
        Session = sessionmaker(bind=engine)
        session = Session()

        holding = session.query(SimulatedHolding) \
            .filter(SimulatedHolding.order_id == buy_order_id)[0]
        holding.status = 'SOLD'
        
        if product_id != "USDC-USDC":
            holding_usdc = session.query(SimulatedHolding).filter(
                SimulatedHolding.coinbase_product_id == "USDC-USDC")[0]
            total_sale_amount_usdc = limit_price * base_size
            fee = total_sale_amount_usdc * TAKER_FEE
            holding_usdc.quantity = holding_usdc.quantity + total_sale_amount_usdc - fee

        session.commit()
        session.close()

    def portfolio(self):
        Session = sessionmaker(bind=engine)
        session = Session()

        simulated_holdings = session.query(SimulatedHolding) \
            .filter(SimulatedHolding.status == 'CURRENT') \
            .filter(SimulatedHolding.coinbase_product_id != "USDC-USDC")
        holdings = list(map(lambda h: Holding(
            h.coinbase_product_id, h.quantity), simulated_holdings))
        session.close()

        if len(holdings) > 0:
            currencies = set(map(lambda h: h.product_id, holdings))
            bids = self.get_best_bids(currencies)

            total_usdc = 0
            for holding in holdings:
                holding.set_best_bid(bids[holding.product_id])
                total_usdc += holding.balance_usd

            for holding in holdings:
                holding.set_allocation(
                    round(holding.balance_usd / total_usdc, 2))

        return holdings

    def holdings_usdc(self):
        Session = sessionmaker(bind=engine)
        session = Session()

        simulated_holdings = session.query(SimulatedHolding) \
            .filter(SimulatedHolding.status == 'CURRENT') \
            .filter(SimulatedHolding.coinbase_product_id == "USDC-USDC")
        
        holdings = list(map(lambda h: Holding(
            h.coinbase_product_id, h.quantity), simulated_holdings))
        session.close()

        return holdings
    def reset(self):
        Session = sessionmaker(bind=engine)
        session = Session()

        # Delete all records from the SimulatedHoldings table
        session.query(SimulatedHolding).delete()

        # Commit the changes and close the session
        session.commit()
        session.close()

    def get_best_bids(self, product_ids):
        product_ids = filter(lambda p: p != "USDC-USDC", product_ids)
        prices = self.coinbase_client.get_best_bid_ask(product_ids)

        bids = {
            'USDC-USDC': 1.0
        }
        for pricebook in prices.pricebooks:
            bids[pricebook.product_id] = float(pricebook.bids[0].price)
        return bids

    def get_best_asks(self, product_ids):
        product_ids = filter(lambda p: p != "USDC-USDC", product_ids)
        prices = self.coinbase_client.get_best_bid_ask(product_ids)

        bids = {
            'USDC-USDC': 1.0
        }
        for pricebook in prices.pricebooks:
            bids[pricebook.product_id] = float(pricebook.asks[0].price)
        return bids


        # coinbase_order = client.create_limit_order(
        #     client_order_id=client_order_id,
        #     product_id=order.coinbase_product_id,
        #     side=Side.SELL,
        #     limit_price=exchange_rate_usd,
        #     base_size=order.quantity)
        # if coinbase_order.order_error:
        #     update_order_status(order.id, "SALE_ERROR")
        #     error_message = coinbase_order.order_error.message if coinbase_order.order_error.message else "Unknown error"
        #     logging.error("Failed to create and execute sell order for order ID %s: %s", order.id, error_message)
        # else:
        #     update_order_status(order.id, "SOLD")
        #     logging.info("Sell order executed successfully for order ID %s", order.id)