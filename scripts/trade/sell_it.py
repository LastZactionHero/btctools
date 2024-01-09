from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
import requests
import csv
import sys
import time
import os
import datetime
from dotenv import load_dotenv

# Constants
CSV_FIELDNAMES = ['STATUS', 'ACTION', 'COINBASE_PRODUCT_ID', 'TRIGGER', 'TRIGGER_PRICE_USD', 'SELL_PRICE_USD', 'QUANTITY']
FLOAT_FIELDS = ['TRIGGER_PRICE_USD', 'SELL_PRICE_USD', 'QUANTITY']
DATE_FORMAT = "%Y%m%d%H%M%S"
TIME_SLEEP_SECONDS = 60

# Load environment variables
load_dotenv()
api_key_name = os.getenv("COINBASE_API_KEY_NAME")
private_key = os.getenv("COINBASE_PRIVATE_KEY")
client = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(api_key_name, private_key)

def load_orders_from_csv(filename):
    field_mapping = {field: field.lower() for field in CSV_FIELDNAMES}
    orders = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            order = {field_mapping[field]: float(value) if field in FLOAT_FIELDS else value for field, value in row.items()}
            orders.append(order)
    return orders

def write_orders_to_csv(orders, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for order in orders:
            writer.writerow({key.upper(): value for key, value in order.items()})

def build_client_order_id(order):
    symbol = order["coinbase_product_id"].split("-")[0]
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(DATE_FORMAT)
    return f"{order['action']}_{symbol}_{timestamp}"

def print_order_details(order, exchange_rate_usd, client_order_id):
    print(f"{client_order_id}\t{order['coinbase_product_id']}\t bid is ${exchange_rate_usd:08.4f}\t{order['trigger']}\ttrigger at ${order['trigger_price_usd']:08.4f}\tto sell {order['quantity']:08.4f} @ ${order['sell_price_usd']:08.4f}")

def should_trigger_order(order, exchange_rate_usd):
    return ((order['trigger'] == 'PRICE_DROPS_TO' and exchange_rate_usd <= order['trigger_price_usd']) or
            (order['trigger'] == 'PRICE_RAISES_TO' and exchange_rate_usd >= order['trigger_price_usd']))

def main(csv_filename):
    while True:
        orders = load_orders_from_csv(csv_filename)
        print(datetime.datetime.now().strftime("%H:%M:%S"))

        for order in orders:
            if order['status'] != 'OPEN':
                continue

            best_bid_ask = client.get_best_bid_ask([order['coinbase_product_id']])
            exchange_rate_usd = float(best_bid_ask.pricebooks[0].bids[0].price)

            client_order_id = build_client_order_id(order)
            print_order_details(order, exchange_rate_usd, client_order_id)

            if should_trigger_order(order, exchange_rate_usd):
                coinbase_order = client.create_limit_order(
                    client_order_id=client_order_id,
                    product_id=order['coinbase_product_id'],
                    side=Side.SELL,
                    limit_price=exchange_rate_usd,
                    base_size=order['quantity'])
                print(coinbase_order)
                order['status'] = 'TRIGGERED'

        write_orders_to_csv(orders, csv_filename)
        time.sleep(TIME_SLEEP_SECONDS)

if __name__ == "__main__":
    csv_filename = sys.argv[1]
    main(csv_filename)
