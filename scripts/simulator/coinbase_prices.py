class CoinbasePrices:
    def __init__(self, ask_data, bid_data, timesource):
        self.ask_data = ask_data
        self.bid_data = bid_data
        self.timesource = timesource

    def ask(self, symbol):
         row_idx = self.ask_data['timestamp'].searchsorted(self.timesource.now()) - 1
         return self.ask_data.iloc[row_idx][symbol]
    
    def bid(self, symbol):
         row_idx = self.bid_data['timestamp'].searchsorted(self.timesource.now()) - 1
         return self.bid_data.iloc[row_idx][symbol]