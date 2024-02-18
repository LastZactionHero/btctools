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

DB_FILENAME = "./db/live.db"
FILENAME_CRYPTO_EXCHANGE_RATES = "./data/crypto_exchange_rates.csv"
CRYPTO_EXCHANGE_RATES_URL = "http://144.202.24.235/crypto_exchange_rates.csv"
MAX_CSV_LATENCY_MIN = 5

load_dotenv()

# TODO:
# Remove prediction flitering

def buy(buyer):
    fetcher = CryptoExchangeRatesFetcher(CRYPTO_EXCHANGE_RATES_URL, FILENAME_CRYPTO_EXCHANGE_RATES)

    data_crypto_exchange_rates = fetcher.fetch(cached=False)
    data_crypto_exchange_rates = pd.read_csv(FILENAME_CRYPTO_EXCHANGE_RATES)
    
    # Check the last timestamp in the DataFrame
    last_timestamp = int(data_crypto_exchange_rates['timestamp'].iloc[-1])
    
    # Get the current timestamp using timesource.now()
    current_timestamp = timesource.now()
    
    # Calculate the time difference in minutes
    time_diff_minutes = (current_timestamp - last_timestamp) / 60

    if time_diff_minutes <= MAX_CSV_LATENCY_MIN:
        buyer.buy(data_crypto_exchange_rates, latest=True)
    else:
        print(f"Error: Data timestamp too old: {time_diff_minutes} minutes")



engine = init_db_engine(DB_FILENAME)
context = {
    "buy_interval_minutes": 5,
    "run_duration_minutes": None,  # 7 * 24 * 60, # 5 days
    "raise_stoploss_threshold": 1.018,  # Sweep 3
    "sell_stoploss_floor": 0.00184,  # Sweep 1
    "order_amount_usd": 50.0,
    "stop_loss_percent": 0.0782,  # Sweep 2
    "take_profit_percent": 1.1,
    "max_delta": 4.3012,  # Sweep 4
    "max_spread": 1.0,  # Sweep 5
    "sell_all_on_hit": False,
    "loss_recovery_after_minutes": 4 * 24 * 60,
    "single_buy": True,
    "engine": engine,
}
Base.metadata.create_all(context["engine"])

timesource = Timesource()
broker = Broker()
seller = Seller(context=context, broker=broker, timesource=timesource)

buyer_prediction_model = BuyerPredictionModel(timesource)
buyer = Buyer(context=context, model=buyer_prediction_model, broker=broker, timesource=timesource)

last_buy_timestamp = 0

while True:
    try:
        print(f"{timesource.now()}")
        print(f"${broker.usdc_available()} USDC")

        if last_buy_timestamp == 0 or ((timesource.now() - last_buy_timestamp) > context[
            "buy_interval_minutes"
        ] * 60):
            last_buy_timestamp = timesource.now()
            if broker.usdc_available() > context['order_amount_usd']:
                buy(buyer)
            else:
                print("Out of money!")

        seller.sell()
    except Exception as e:
        print(e)
    time.sleep(10)
