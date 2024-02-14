import os
import pandas as pd
from datetime import datetime
from scripts.simulator.buyer import Buyer
from scripts.simulator.prediction_cache import PredictionCache
from scripts.simulator.seller import Seller
from scripts.simulator.buyer_prediction_model import BuyerPredictionModel
from scripts.simulator.coinbase_prices import CoinbasePrices
from scripts.db.models import init_db_engine, Base

# DB_FILENAME = "./db/magpie.db"

# Load crypto data, ask, bids csv
FILENAME_CRYPTO_EXCHANGE_RATES = "./data/crypto_exchange_rates.csv"
FILENAME_ASKS = "./data/asks.csv"
FILENAME_BIDS = "./data/bids.csv"

data_crypto_exchange_rates = pd.read_csv(FILENAME_CRYPTO_EXCHANGE_RATES)
data_asks = pd.read_csv(FILENAME_ASKS)
data_bids = pd.read_csv(FILENAME_BIDS)

experiment_name = "sell_all_on_hit"
parameter_values = [True, False]

def time_remaining(start_time, step_idx, step_count):
    now = datetime.now()

    percent_complete =  step_idx / step_count
    if percent_complete <= 0:
        return "0%"

    time_remaining = (now - start_time) / percent_complete * (1 - percent_complete)

    hours = time_remaining.seconds // 3600
    minutes = (time_remaining.seconds % 3600) // 60
    seconds = time_remaining.seconds % 60

    return f"{round(percent_complete * 100)}%: {hours:02d}:{minutes:02d}:{seconds:02d} remaining"

for parameter_value in parameter_values:
    parameter_str = None
    if type(parameter_value) == bool:
        if parameter_value == True:
            parameter_str = "1_0"
        else:
            parameter_str = "0_0"
    else: 
        parameter_str = str(parameter_value).replace(".", "_")

    db_filename = f"./db/exp_{experiment_name}_{parameter_str}.db"
   
    context = {
        "buy_interval_minutes": 10,
        "run_duration_minutes": None,
        "raise_stoploss_threshold": 1.018, # Sweep 3
        "sell_stoploss_floor": 0.002, # Sweep 1
        "order_amount_usd": 100.0,
        "stop_loss_percent": 0.08, # Sweep 2
        "take_profit_percent": 1.1,
        "max_delta": 4.5, # Sweep 4
        "max_spread": 1.0, # Sweep 5
        "sell_all_on_hit": False,
        "engine": init_db_engine(db_filename)
    }
    Base.metadata.create_all(context['engine']) 

    prices = CoinbasePrices(data_asks, data_bids)
    prediction_cache = PredictionCache()
    buyer_prediction_model = BuyerPredictionModel(data_crypto_exchange_rates, cache=prediction_cache)
    buyer = Buyer(context=context, data=data_crypto_exchange_rates, model=buyer_prediction_model, prices=prices)
    seller = Seller(context=context, prices=prices)

    start_timestamp = int(data_asks.iloc[0]['timestamp'])

    start_time = datetime.now()
    last_buy_timestamp = 0
    for step, timestamp in enumerate(data_asks['timestamp'].values):
        os.system('clear')
        print(f"Step {step}/{len(data_asks['timestamp'].values)}")
        print(time_remaining(start_time, step, len(data_asks['timestamp'].values)))

        current_timestamp = int(timestamp)
        seller.sell(current_timestamp)

        if (timestamp - last_buy_timestamp) > context['buy_interval_minutes'] * 60:
            last_buy_timestamp = timestamp
            buyer.buy(timestamp)

        if (context['run_duration_minutes'] and current_timestamp > start_timestamp + context['run_duration_minutes'] * 60):
            break
