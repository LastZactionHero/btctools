class CoinbasePrices:
    def __init__(self, ask_data, bid_data):
        self.ask_data = ask_data
        self.bid_data = bid_data

    def ask_at_timestamp(self, symbol, timestamp):
         row_idx = self.ask_data['timestamp'].searchsorted(timestamp) - 1
         return self.ask_data.iloc[row_idx][symbol]
    
    def bid_at_timestamp(self, symbol, timestamp):
         row_idx = self.bid_data['timestamp'].searchsorted(timestamp) - 1
         return self.bid_data.iloc[row_idx][symbol]