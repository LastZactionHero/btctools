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
from scripts.db.models import Order, sessionmaker
from datetime import datetime

load_dotenv()

broker = Broker()

print(broker.usdc_available())

# broker.buy("PREP_001", "POLS-USDC", 1, 0.8855)
# broker.sell("PREP_002", "BADGER-USDC", 1, 4.36)
# prices = broker.prices()
# ask = prices.ask("BTRST")
# ask = 0.915556923992
# x = broker.buy("PREP_007", "BTRST-USDC", 5.102022011, ask)
# print(x)
DB_FILENAME = "./db/live.db"
engine = init_db_engine(DB_FILENAME)
Session = sessionmaker(bind=engine)
session = Session()
order = Order(
    order_id="PREP_007",
    coinbase_product_id="BTRST-USDC",
    quantity=11.02,
    purchase_price=0.0918,
    status="OPEN",
    action="SELL",
    stop_loss_percent=0.0782,
    profit_percent=1.1,
    predicted_max_delta=0.0,
    predicted_min_delta=0.0,
    purchase_time_spread_percent=0.0,
    created_at=datetime.now()
)
session.add(order)
session.commit()
session.close()
