import pandas as pd
import time
import logging
from dotenv import load_dotenv
import json
import datetime
from scripts.data_collection import coingecko_csv_updater
from scripts.data_collection.coingecko_csv_updater import CoingeckoCsvUpdater
from scripts.trade.buyer import Buyer
from scripts.trade.seller import Seller
from scripts.trade.buyer_prediction_model import BuyerPredictionModel
from scripts.live.broker import Broker
from scripts.live.timesource import Timesource
from scripts.live.full_coingecko_csv_fetcher import FullCoingeckoCSVFetcher
from scripts.db.models import init_db_engine, Base, Order
from scripts.trade.dump_status_info import DumpStatusInfo
from sqlalchemy.orm import sessionmaker
import json
import copy

DB_FILENAME = "./db/live.db"
FILENAME_CRYPTO_EXCHANGE_RATES = "./data/crypto_exchange_rates.csv"
FILENAME_STATUS_JSON = "./logs/status.json"
FILENAME_STATUS_HTML = "./logs/status.html"
CRYPTO_EXCHANGE_RATES_URL = "http://144.202.24.235/crypto_exchange_rates.csv"
MAX_CSV_LATENCY_MIN = 5

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="./logs/live.log",  # Example path on Debian
    filemode="a",
)  # Append mode

# Creating logger object
logger = logging.getLogger("LiveLogger")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

load_dotenv()


def buy(buyer):
    data_crypto_exchange_rates = pd.read_csv(FILENAME_CRYPTO_EXCHANGE_RATES)

    # Check the last timestamp in the DataFrame
    last_timestamp = int(data_crypto_exchange_rates["timestamp"].iloc[-1])

    # Get the current timestamp using timesource.now()
    current_timestamp = timesource.now()

    # Calculate the time difference in minutes
    time_diff_minutes = (current_timestamp - last_timestamp) / 60

    if time_diff_minutes <= MAX_CSV_LATENCY_MIN:
        buyer.buy(data_crypto_exchange_rates, latest=True)
    else:
        logger.error(f"Data timestamp too old: {time_diff_minutes} minutes")


engine = init_db_engine(DB_FILENAME)
context = {
    "buy_interval_minutes": 5,
    "run_duration_minutes": None,
    "raise_stoploss_threshold": 1.018,
    "sell_stoploss_floor": 0.00184,
    "order_amount_usd": 50.0,
    "stop_loss_percent": 0.0782,
    "take_profit_percent": 1.1,
    "max_delta": 2.5, # 4.3012 default
    "max_spread": 1.0,
    "sell_all_on_hit": False,
    "loss_recovery_after_minutes": 4 * 24 * 60,
    "single_buy": True,
    "live_trades": True,
    "engine": engine,
}
Base.metadata.create_all(context["engine"])

timesource = Timesource()
broker = Broker(logger=logger, context=context)
seller = Seller(context=context, broker=broker, timesource=timesource, logger=logger)

buyer_prediction_model = BuyerPredictionModel(timesource, logger=logger)
buyer = Buyer(
    context=context,
    model=buyer_prediction_model,
    broker=broker,
    timesource=timesource,
    logger=logger,
)
last_buy_timestamp = 0

fetcher = FullCoingeckoCSVFetcher(
    CRYPTO_EXCHANGE_RATES_URL, FILENAME_CRYPTO_EXCHANGE_RATES, logger
)
data_crypto_exchange_rates = fetcher.fetch(cached=False)
last_coingecko_timestamp = timesource.now()

coingecko_csv_updater = CoingeckoCsvUpdater(
    timesource, FILENAME_CRYPTO_EXCHANGE_RATES, logger
)

while True:
    try:
        logger.info(f"{timesource.now()}")
        usdc_available = broker.usdc_available()
        logger.info(f"${usdc_available} USDC")

        if last_coingecko_timestamp == 0 or (
            (timesource.now() - last_coingecko_timestamp) > 60
        ):
            coingecko_csv_updater.fetch_and_update()
            last_coingecko_timestamp = timesource.now()

        if last_buy_timestamp == 0 or (
            (timesource.now() - last_buy_timestamp)
            > context["buy_interval_minutes"] * 60
        ):
            last_buy_timestamp = timesource.now()
            if broker.usdc_available() > context["order_amount_usd"]:
                buy(buyer)
            else:
                logger.info("Out of money!")

        seller.sell()

        status_dump = DumpStatusInfo(FILENAME_STATUS_JSON, FILENAME_STATUS_HTML)
        status_dump.save_status_info(
            timesource,
            last_buy_timestamp,
            last_coingecko_timestamp,
            usdc_available,
            broker.prices(),
            broker.holdings(),
            context,
        )
    except Exception as e:
        logger.error(e)
    time.sleep(2)
