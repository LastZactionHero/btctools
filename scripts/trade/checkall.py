import datetime
import logging
import logging_setup
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

openai_api_key = os.getenv("OPENAI_API_KEY")
api_key_name = os.getenv("COINBASE_API_KEY_NAME")
private_key = os.getenv("COINBASE_PRIVATE_KEY")

coinbaseClient = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(
    api_key_name, private_key
)
broker = SimulatedBroker(coinbaseClient)

for key in gecko_coinbase_currency_map.keys():
    symbol = gecko_coinbase_currency_map[key]
    pid = f"{symbol}-USDC"
    try:
        broker.get_best_asks([pid])
    except:
        print(f"FAILED: {symbol} {pid}")