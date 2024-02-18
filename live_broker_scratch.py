import pandas as pd
import time
from dotenv import load_dotenv
from scripts.trade.buyer import Buyer
from scripts.trade.seller import Seller
from scripts.trade.buyer_prediction_model import BuyerPredictionModel
from scripts.live.broker import Broker
from scripts.live.timesource import Timesource
from scripts.live.crypto_exchange_rates_fetcher import CryptoExchangeRatesFetcher
from scripts.db.models import init_db_engine, Base


load_dotenv()

broker = Broker()

print(broker.usdc_available())

# broker.buy("PREP_001", "POLS-USDC", 1, 0.8855)
# broker.sell("PREP_002", "BADGER-USDC", 1, 4.36)
prices = broker.prices()
best_bid = prices.bid("LRC")
x = broker.sell("PREP_005", "LRC-USDC", 100000, best_bid)
print(x)