import datetime
import logging
import traceback
import logging_setup
import numpy as np
import os
import pandas as pd
import re
import requests
import time

from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
from coingecko_coinbase_pairs import gecko_coinbase_currency_map
from gpt import GPT
from prettytable import PrettyTable
from print_utils import portfolio_table
from simulated_broker import SimulatedBroker
from db.models import engine, Order, sessionmaker
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

TIME_SLEEP_SECONDS = 5 * 60
ASK_PRICE_ADJ = 1.00

POSITIVE_PREDICTION_MIN_DELTA = 1.1
MIN_POSITIVE_PREDICTIONS = 1

CSV_DIR = "./data"
EXCHANGE_RATE_CSV_FILENAME = "./data/crypto_exchange_rates.csv"
PREDICTIONS_CSV_FILENAME = "./data/lstm_predictions.csv"
EXCHANGE_RATE_CSV_URL = "http://144.202.24.235/crypto_exchange_rates.csv"
PREDICTIONS_CSV_URL = "http://144.202.24.235/lstm_predictions.csv"
MODEL_FILE = "./models/lstm_series_240m.h5"

MAX_BUY_AMOUNT_USDC = 100
MIN_BUY_AMOUNT_USDC = 50

STOP_LOSS_PERCENT = 0.96
PROFIT_PERCENT = 1.1

TIME_ABOVE_MAX_PERCENTAGE = 1.0
TIME_ABOVE_MIN_PERCENTAGE = 0.0

# MIN_HIT_COUNT = 2
# HIT_FACTOR = 1.04

SEQUENCE_INTERVAL = 1
SEQUENCE_LOOKBEHIND_MINUTES = 240 #7 * 24 * 60
PREDICTION_LOOKAHEAD_MINUTES = 30

logging_setup.init_logging("buy.log")

openai_api_key = os.getenv("OPENAI_API_KEY")
api_key_name = os.getenv("COINBASE_API_KEY_NAME")
private_key = os.getenv("COINBASE_PRIVATE_KEY")

coinbaseClient = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(
    api_key_name, private_key
)
broker = SimulatedBroker(coinbaseClient)


def hit_counts_in_range(coin, data, prior_minutes):
    prior_values = data[coin][len(data) - prior_minutes : len(data) - 1].values

    hit_count = 0
    skip_until = 0

    for idx, value in enumerate(prior_values):
        if idx < skip_until:
            continue

        next_values = prior_values[idx:]
        max_subsequence_gain = (next_values.max() - value) / value

        if max_subsequence_gain < 0.04:
            continue

        for next_idx, next_value in enumerate(next_values):
            gain = (next_value - value) / value
            if gain > 0.04:
                skip_until = idx + next_idx
                hit_count += 1
                break
    return hit_count


def percent_above_hit_percent(coin, data, prior_minutes):
    prior_values = data[coin][len(data) - prior_minutes : len(data) - 1].values
    latest_price = prior_values[-1]
    minutes_above_hit_price = len(prior_values[prior_values > latest_price * HIT_FACTOR])

    return minutes_above_hit_price / prior_minutes


def build_positive_predictions(predictions, historical_data):
    logging.info(predictions)
    positive_predictions = predictions[
        predictions["Max Delta"] > POSITIVE_PREDICTION_MIN_DELTA
    ]

    columns = [
        "Symbol",
        "Coin",
        "Latest Price",
        "Predicted Delta",
        "Hit Count",
        "Min Above",
    ]
    df = pd.DataFrame(columns=columns)

    for index, prediction in positive_predictions.iterrows():
        coin = prediction.iloc[0]
        
        if coin not in gecko_coinbase_currency_map.keys():
            continue

        symbol = gecko_coinbase_currency_map[coin]
        if symbol == "UNSUPPORTED":
            continue

        # hit_count = hit_counts_in_range(coin, historical_data, PRIOR_MINUTES_TO_REVIEW)
        # percent_above = percent_above_hit_percent(
        #     coin, historical_data, PRIOR_MINUTES_TO_REVIEW
        # )

        row = [
            gecko_coinbase_currency_map[coin],
            coin,
            historical_data[coin].values[-1],
            prediction["Max Delta"],
            10,
            0.5,
        ]
        df.loc[len(df)] = row

    logging.info("Pre-Filter Predictions:")
    logging.info(df)
    # df_filtered = df[(df["Min Above"] >= TIME_ABOVE_MIN_PERCENTAGE) & (df["Min Above"] <= TIME_ABOVE_MAX_PERCENTAGE)]
    df_filtered = df[(df["Predicted Delta"] >= POSITIVE_PREDICTION_MIN_DELTA)]
    # df_filtered = df_filtered[(df_filtered["Hit Count"] >= MIN_HIT_COUNT)]
    # df_filtered = df_filtered.sort_values(by="Min Above")
    logging.info("Post-Filter Predictions:")
    logging.info(df_filtered)
    return df_filtered


def predictions_table(predictions):
    table = PrettyTable(["Symbol", "Spread %", "# 4% Runs", "Time Above 4%"])
    table.align["Symbol"] = "l"
    table.align["Spread %"] = "r"
    table.align["# 4% Runs"] = "r"
    table.align["Time Above 4%"] = "r"
    for index, prediction in predictions.iterrows():
        table.add_row(
            [
                prediction["Symbol"],
                f"{prediction['Spread']}%",
                prediction["Hit Count"],
                "{:.2f}%".format(prediction["Min Above"] * 100.0),
            ]
        )
    return table


def build_purchase_decision_prompt(predictions, portfolio, balance_usdc):
    prompt = ""
    prompt += "PREDICTIONS:\n"
    prompt += str(predictions_table(predictions)) + "\n"
    prompt += "Anything shown here is appropriate to buy\n\n"
    prompt += "- Symbol: The Coinbase ticker symbol\n"
    prompt += "- Spread: Current percentage delta between best bid and ask price. Prefer lower spreads.\n"
    prompt += "- # 4% Runs: How many times has this coin gone up 4% in the last 14 days. A signal the coin is volitile enough to cover the fees.\n"
    prompt += "- Time Above 4%: In the last two weeks, what percentage of time has the value been over 4%? Prefer values around 50%.\n"
    prompt += "\n"
    prompt += "CURRENT PORTFOLIO:\n"
    prompt += str(portfolio_table(portfolio)) + "\n\n"
    prompt += f"USD AVAILABLE TO INVEST: ${balance_usdc}"
    prompt += "\n"
    prompt += "You are part of an automated cryptocurrency trading system tasked with evaluating the best currency to buy.\n"
    prompt += "Select a coin from PREDICTIONS to purchase next. Try to:\n"
    prompt += "- Maintain a balanced portfolio, preferring new coins and avoiding allocation over 10%\n"
    prompt += "- It's okay to decline with symbol NONE if the portfolio will become too unbalanced- more picks will come later.\n"
    prompt += "\n"
    prompt += "Select ONE symbol for this purchase, or NONE.\n"
    prompt += "Format your answer as: PURCHASE[SYMBOL]"
    return prompt


def parse_llm_purchase_response(response):
    pattern = r"PURCHASE\[(.*?)\]"
    match = re.search(pattern, response)
    if match:
        symbol = match.group(1)
    else:
        symbol = ""
    return symbol


def build_client_order_id(symbol):
    DATE_FORMAT = "%Y%m%d%H%M%S"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(DATE_FORMAT)
    return f"BUY_{symbol}_{timestamp}"


def add_order(order_id, coinbase_product_id, quantity, purchase_price, spread, predictions):
    Session = sessionmaker(bind=engine)
    session = Session()

    symbol = coinbase_product_id.split('-')[0]
    coinbase_gecko_currency_map = {value: key for key, value in gecko_coinbase_currency_map.items()}
    coin_name = coinbase_gecko_currency_map[symbol]
    coin_prediction = predictions[predictions['Coin'] == coin_name].iloc[0]

    predictions['Max Delta'].values.mean()
    predictions['Max Delta'].values.std()
    predictions['Min Delta'].values.mean()
    predictions['Min Delta'].values.std()

    order = Order(
        order_id=order_id,
        coinbase_product_id=coinbase_product_id,
        quantity=quantity,
        purchase_price=purchase_price,
        status="OPEN",
        action="SELL",
        stop_loss_percent=STOP_LOSS_PERCENT,
        profit_percent=PROFIT_PERCENT,
        predicted_max_delta=coin_prediction['Max Delta'],
        predicted_min_delta=coin_prediction['Min Delta'],
        num_predictions_over_hit=len(predictions[predictions['Max Delta'] > POSITIVE_PREDICTION_MIN_DELTA]),
        max_delta_average=predictions['Max Delta'].values.mean(),
        max_delta_std=predictions['Max Delta'].values.std(),
        min_delta_average=predictions['Min Delta'].values.mean(),
        min_delta_std=predictions['Min Delta'].values.std(),
        purchase_time_spread_percent=spread,
        created_at=datetime.datetime.now(),  # Add the timestamp here
    )
    session.add(order)
    session.commit()
    logging.info(f"Order added to database with ID: {order.id}")
    session.close()


def fetch_files():
    try:
        # Check if the CSV directory exists, and create it if not
        if not os.path.exists(CSV_DIR):
            os.makedirs(CSV_DIR)

        # Remove existing files if they exist
        if os.path.exists(EXCHANGE_RATE_CSV_FILENAME):
            os.remove(EXCHANGE_RATE_CSV_FILENAME)
        # Download exchange rate CSV
        response = requests.get(EXCHANGE_RATE_CSV_URL)
        response.raise_for_status()

        with open(EXCHANGE_RATE_CSV_FILENAME, "wb") as file:
            file.write(response.content)

        logging.info("CSV files downloaded successfully.")
    except Exception as e:
        logging.error(f"Error: {str(e)}")


def build_predictions(data):
    model = load_model(MODEL_FILE)
    predict_sequences = data.loc[list(range(len(data) - SEQUENCE_LOOKBEHIND_MINUTES, len(data-1), 1))]

    coins = []
    scalers = []
    predict_sequences_scaled = []

    for i, coin in enumerate(predict_sequences):
        scaler = MinMaxScaler()
        ps = scaler.fit_transform(predict_sequences[coin].values.reshape(1,-1).T).flatten()
        
        coins.append(coin)
        scalers.append(scaler)
        predict_sequences_scaled.append(ps)

    predictions_scaled = model.predict(np.array(predict_sequences_scaled))

    predictions = pd.DataFrame(columns=['Coin', 'Latest', 'Max', 'Max Delta', 'Min', 'Min Delta'])
    for i, ps in enumerate(predictions_scaled):
        latest_price = data.iloc[-1,i]

        unscaled = scalers[i].inverse_transform(ps.reshape(1, -1).T).flatten()
        p_max = unscaled.max()
        p_min = unscaled.min()

        d_max = round((p_max - latest_price) / latest_price * 100.0, 2)
        d_min = round((p_min - latest_price) / latest_price * 100.0, 2)

        row = pd.DataFrame({
            'Coin': [coins[i]],
            'Latest': latest_price,
            'Max': [p_max],
            'Max Delta': [d_max],
            'Min': [p_min],
            'Min Delta': [d_min]
        })
        predictions = pd.concat([predictions, row])

    predictions = predictions.sort_values(by="Max Delta", ascending=False)
    return predictions

def fetch_current_prices(positive_predictions):
    symbols = positive_predictions['Symbol'].values
    product_ids = list(map(lambda s: f"{s}-USDC", symbols))
                                   
    bids = broker.get_best_bids(product_ids)
    asks = broker.get_best_asks(product_ids)

    spreads = []
    for product_id in product_ids:
        bid = bids[product_id]
        ask = asks[product_id]
        spreads.append(round((ask - bid) / bid * 100, 2))
    positive_predictions['Spread'] = spreads
    return positive_predictions

def runit():
    fetch_files()

    historical_data = pd.read_csv(EXCHANGE_RATE_CSV_FILENAME)
    predictions = build_predictions(historical_data)
    
    positive_predictions = build_positive_predictions(predictions, historical_data)
    if len(positive_predictions) == 0:
        logging.info("No positive predictions")
        return
    positive_predictions = fetch_current_prices(positive_predictions)

    logging.info('positive predictions')
    logging.info(positive_predictions)
    if len(positive_predictions) >= MIN_POSITIVE_PREDICTIONS:
        portfolio = broker.portfolio()
        holdings_usdc = broker.holdings_usdc()
        buy_amount_usdc = min(holdings_usdc[0].balance_usd, MAX_BUY_AMOUNT_USDC)
        if buy_amount_usdc > MIN_BUY_AMOUNT_USDC:
            prediction_prompt = build_purchase_decision_prompt(
                positive_predictions, portfolio, holdings_usdc[0].balance_usd
            )

            for purchase_decision_symbol in positive_predictions['Symbol'].values:
                product_id = f"{purchase_decision_symbol}-USDC"
                limit_price = broker.get_best_asks([product_id])[product_id]
                buy_amount_usdc = min(holdings_usdc[0].balance_usd, MAX_BUY_AMOUNT_USDC)
                buy_order = broker.buy(
                    build_client_order_id(purchase_decision_symbol),
                    product_id,
                    limit_price * ASK_PRICE_ADJ,
                    buy_amount_usdc,
                )

                spread = positive_predictions[positive_predictions['Symbol'] == purchase_decision_symbol].iloc[0]['Spread']
                add_order(buy_order['order_id'],
                        product_id,
                        buy_order["quantity"],
                        buy_order["purchase_price"],
                        spread,
                        predictions)

                logging.info(portfolio_table(broker.portfolio()))
                logging.info(portfolio_table(broker.holdings_usdc()))
            
        else:
            logging.info("Out of money...")
    else:
        logging.info("Nothing good to buy...")
        


while True:
    try:
        runit()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
    time.sleep(TIME_SLEEP_SECONDS)