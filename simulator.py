import pandas as pd
from scripts.simulator.buyer import Buyer
from scripts.simulator.seller import Seller
from scripts.simulator.buyer_prediction_model import BuyerPredictionModel
from scripts.simulator.coinbase_prices import CoinbasePrices

BUY_INTERVAL_MINUTES = 10
RUN_DURATION_MINUTES = None #24 * 60

# Load crypto data, ask, bids csv
FILENAME_CRYPTO_EXCHANGE_RATES = "./data/crypto_exchange_rates.csv"
FILENAME_ASKS = "./data/asks.csv"
FILENAME_BIDS = "./data/bids.csv"

data_crypto_exchange_rates = pd.read_csv(FILENAME_CRYPTO_EXCHANGE_RATES)
data_asks = pd.read_csv(FILENAME_ASKS)
data_bids = pd.read_csv(FILENAME_BIDS)

prices = CoinbasePrices(data_asks, data_bids)
buyer_prediction_model = BuyerPredictionModel(data_crypto_exchange_rates)
buyer = Buyer(data=data_crypto_exchange_rates, model=buyer_prediction_model, prices=prices)
seller = Seller(prices=prices)

start_timestamp = int(data_asks.iloc[0]['timestamp'])

last_buy_timestamp = 0
for timestamp in data_asks['timestamp'].values:
    current_timestamp = int(timestamp)
    seller.sell(current_timestamp)

    if (timestamp - last_buy_timestamp) > BUY_INTERVAL_MINUTES * 60:
        last_buy_timestamp = timestamp
        buyer.buy(timestamp)

    if (RUN_DURATION_MINUTES and current_timestamp > start_timestamp + RUN_DURATION_MINUTES * 60):
        break

    