import json
import os
import pandas as pd
import random
from datetime import datetime
from scripts.simulator.buyer import Buyer
from scripts.simulator.prediction_cache import PredictionCache
from scripts.simulator.seller import Seller
from scripts.simulator.buyer_prediction_model import BuyerPredictionModel
from scripts.simulator.coinbase_prices import CoinbasePrices
from scripts.simulator.genetic_context_modifier import GeneticContextModifier
from scripts.simulator.run_results import RunResults
from scripts.db.models import init_db_engine, Base

GENETIC_CONTEXT_MODIFIER = False
RANDOM_START_TIMESTAMP = True

# Load crypto data, ask, bids csv
FILENAME_CRYPTO_EXCHANGE_RATES = "./data/crypto_exchange_rates.csv"
FILENAME_ASKS = "./data/asks.csv"
FILENAME_BIDS = "./data/bids.csv"

data_crypto_exchange_rates = pd.read_csv(FILENAME_CRYPTO_EXCHANGE_RATES)
data_asks = pd.read_csv(FILENAME_ASKS)
data_bids = pd.read_csv(FILENAME_BIDS)

experiment_name = "recovery_mode"
parameter_values = [4, 5, 6, 7, 8, 9, 10, 20]

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


genetic_context_modifier = GeneticContextModifier()

for parameter_value in parameter_values:
    genetic_context_modifier.mutate_context()

    parameter_str = None
    if type(parameter_value) == bool:
        if parameter_value == True:
            parameter_str = "1_0"
        else:
            parameter_str = "0_0"
    elif type(parameter_value) == int:
        parameter_str = f"{parameter_value}_0"
    else: 
        parameter_str = str(parameter_value).replace(".", "_")

    db_filename = f"./db/exp_{experiment_name}_{parameter_str}.db"
   
    engine = init_db_engine(db_filename)
    base_context = {
        "buy_interval_minutes": 10,
        "run_duration_minutes": 5 * 24 * 60, # 5 days
        "raise_stoploss_threshold": 1.018, # Sweep 3
        "sell_stoploss_floor": 0.00184, # Sweep 1
        "order_amount_usd": 100.0,
        "stop_loss_percent": 0.0782, # Sweep 2
        "take_profit_percent": 1.1,
        "max_delta": 4.3012, # Sweep 4
        "max_spread": 1.0, # Sweep 5
        "sell_all_on_hit": False,
        "loss_recovery_after_minutes": parameter_value * 24 * 60,
        "engine": engine
    }

    context = base_context.copy()
    if GENETIC_CONTEXT_MODIFIER:
        context.update(genetic_context_modifier.trial_context['context'])
    print(context)

    Base.metadata.create_all(context['engine']) 

    prices = CoinbasePrices(data_asks, data_bids)
    prediction_cache = PredictionCache()
    buyer_prediction_model = BuyerPredictionModel(data_crypto_exchange_rates, cache=prediction_cache)
    buyer = Buyer(context=context, data=data_crypto_exchange_rates, model=buyer_prediction_model, prices=prices)
    seller = Seller(context=context, prices=prices)

    start_timestamp = int(data_asks.iloc[0]['timestamp'])
    if RANDOM_START_TIMESTAMP:
        start_timestamp = None
        while start_timestamp is None:
            start_idx = random.randrange(len(data_asks))
            proposed_start_timestamp = data_asks.iloc[start_idx]['timestamp']
            end_timestamp = data_asks.iloc[-1]['timestamp']
            minutes_until_data_end = (end_timestamp - proposed_start_timestamp) / 60
            if minutes_until_data_end > context['run_duration_minutes']:
                start_timestamp = proposed_start_timestamp

    start_time = datetime.now()
    last_buy_timestamp = 0
    for step, timestamp in enumerate(data_asks['timestamp'].values):
        if timestamp < start_timestamp:
            continue

        os.system('clear')
        print(f"Step {step}/{len(data_asks['timestamp'].values)}")
        print(time_remaining(start_time, step, len(data_asks['timestamp'].values)))
        print(json.dumps(genetic_context_modifier.trial_context['context'], indent=2))

        current_timestamp = int(timestamp)
        seller.sell(current_timestamp)

        if (timestamp - last_buy_timestamp) > context['buy_interval_minutes'] * 60:
            last_buy_timestamp = timestamp
            buyer.buy(timestamp)

        if (context['run_duration_minutes'] and current_timestamp > start_timestamp + context['run_duration_minutes'] * 60):
            break

    run_results = RunResults(engine, data_bids).get_results()
    print(run_results)
    if GENETIC_CONTEXT_MODIFIER:
        genetic_context_modifier.compare_contexts(percent_change=run_results['percent_change'], total_net=run_results['total_net'])
