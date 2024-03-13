class Broker():
    def __init__(self, prices):
        self.p = prices

    def buy(self, order_id, product_id, amount_usdc, highest_bid):
        return round(amount_usdc / highest_bid, 5)

    def sell(self, order_id, product_id, amount_product, lowest_ask):
        return True

    def prices(self):
        return self.p