# fmt: off
import sys
sys.path.append("./scripts")
from datetime import datetime
from db.models import engine, SimulatedHolding, sessionmaker



class Holding:
    def __init__(self, coinbase_product_id, quantity):
        self.product_id = coinbase_product_id
        self.currency = coinbase_product_id.split('-')[0]
        self.balance_coin = quantity
        self.balance_usd = None
        self.allocation = None

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

    def buy(self, client_order_id, product_id, limit_price, base_size):
        Session = sessionmaker(bind=engine)
        session = Session()

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
            holding_usdc.quantity = holding_usdc.quantity - limit_price * base_size

        session.add(holding)
        session.commit()
        session.close()

    def sell(self, client_order_id, product_id, limit_price, base_size):
        Session = sessionmaker(bind=engine)
        session = Session()

        holding = session.query(SimulatedHolding) \
            .filter(SimulatedHolding.coinbase_product_id == product_id) \
            .filter(SimulatedHolding.quantity == base_size)[0]
        holding.status = 'SOLD'
        
        if product_id != "USDC-USDC":
            holding_usdc = session.query(SimulatedHolding).filter(
                SimulatedHolding.coinbase_product_id == "USDC-USDC")[0]
            holding_usdc.quantity = holding_usdc.quantity + limit_price * base_size

        session.commit()
        session.close()

    def portfolio(self):
        Session = sessionmaker(bind=engine)
        session = Session()

        simulated_holdings = session.query(SimulatedHolding).filter(
            SimulatedHolding.status == 'CURRENT')
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
