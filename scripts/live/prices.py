class Prices:
    def __init__(self, bids, asks):
        self.bids = bids
        self.asks = asks

    def ask(self, symbol):
         return self.asks[f"{symbol}-USDC"]
    
    def bid(self, symbol):
        return self.bids[f"{symbol}-USDC"]