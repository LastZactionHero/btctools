import initpath
import csv
import os
import time
import logging
from dotenv import load_dotenv
from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
from db.models import engine, Order, sessionmaker  
from print_utils import portfolio_table
from simulated_broker import SimulatedBroker
from coingecko_coinbase_pairs import gecko_coinbase_currency_map

# Set up logging
logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)

def get_sorted_keys(symbols):
    symbols.sort()
    return symbols

def prepare_data(asks, symbols):
    data = [asks.get(f"{symbol}-USDC", -1) for symbol in symbols]
    return data

def write_to_csv(symbols, data, path):
    file_exists = os.path.isfile(path)
    with open(path, 'a') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp'] + symbols)
        writer.writerow([int(time.time())] + data)

def main():
    # Load environment variables
    load_dotenv()
    api_key_name = os.getenv("COINBASE_API_KEY_NAME")
    private_key = os.getenv("COINBASE_PRIVATE_KEY")

    try:
        client = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(api_key_name, private_key)
    except Exception as e:
        logging.error("Failed to initialize the Coinbase Advanced Trade API Client: %s", e)
        sys.exit(1)
    broker = SimulatedBroker(client)

    symbols = list(filter(lambda s: s != 'UNSUPPORTED', gecko_coinbase_currency_map.values()))
    products = list(map(lambda s: "{}-USDC".format(s), symbols))

    asks_path = './data/asks.csv'
    bids_path = './data/bids.csv'

    while True:
        try:
            asks = broker.get_best_asks(products)
            bids = broker.get_best_bids(products)

            symbols = get_sorted_keys(symbols)
            asks_data = prepare_data(asks, symbols)
            write_to_csv(symbols, asks_data, asks_path)

            bids_data = prepare_data(bids, symbols)
            write_to_csv(symbols, bids_data, bids_path)
        except Exception as e:
            logging.error("Error occurred: " + str(e))

        time.sleep(10)

if __name__ == "__main__":
    main()