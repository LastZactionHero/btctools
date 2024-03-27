import pandas as pd
import logging
from dotenv import load_dotenv
from scripts.live.broker import Broker
from scripts.live.timesource import Timesource

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


context = {
    "buy_interval_minutes": 2,
    "run_duration_minutes": None,
    "raise_stoploss_threshold": 1.018,
    "sell_stoploss_floor": 0.00184,
    "order_amount_usd": 300.0,
    "stop_loss_percent": 0.0782,
    "take_profit_percent": 1.1,
    "max_delta": 2.0, # 4.3012 default
    "max_spread": 1.0,
    "sell_all_on_hit": False,
    "loss_recovery_after_minutes": 7 * 24 * 60,
    "single_buy": True,
    "live_trades": True,
    "time_above_minutes_to_review": 30 * 24 * 60, # 30 days
    "time_above_threshold": 7.0, # Must have spent at least 7% of prior month above purchase price
    "max_repeat_orders": 2,
    "restricted_coins": []
}

timesource = Timesource()
broker = Broker(logger=logger, context=context)

broker.buy("TEST_01", "NCT-USDC", 5, 0.01)