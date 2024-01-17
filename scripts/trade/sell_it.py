from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
import requests
import csv
import sys
import time
import os
import datetime

# Constants
CSV_FIELDNAMES = ['STATUS','ACTION','COINBASE_PRODUCT_ID','QUANTITY','PURCHASE_PRICE','STOP_LOSS_PERCENT','PROFIT_PERCENT']
FLOAT_FIELDS = ['QUANTITY','PURCHASE_PRICE','STOP_LOSS_PERCENT','PROFIT_PERCENT']
DATE_FORMAT = "%Y%m%d%H%M%S"
TIME_SLEEP_SECONDS = 10
RAISE_STOPLOSS_THRESHOLD = 1.015
RAISE_STOPLOSS_VALUE = 0.99

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
    print(f"{order['coinbase_product_id']}\t bid is ${exchange_rate_usd:08.4f}\tpurchase ${order['purchase_price']:08.4f}\tdelta {delta_since_purchase(order, exchange_rate_usd):0.2f}%\tlow ${stop_loss_price(order):08.4f}\thigh ${profit_price(order):08.4f}")

def stop_loss_price(order):
    return order['purchase_price'] * order['stop_loss_percent']

def profit_price(order):
    return order['purchase_price'] * order['profit_percent']

def delta_since_purchase(order, exchange_rate_usd):
    return 100 * (exchange_rate_usd - order['purchase_price']) / order['purchase_price']

def should_trigger_order(order, exchange_rate_usd):
    return exchange_rate_usd >= profit_price(order) or exchange_rate_usd <= stop_loss_price(order)

def adjust_stoploss(order, exchange_rate_usd):
    prev_value = order['stop_loss_percent']
    next_value = order['stop_loss_percent']

    # if exchange_rate_usd >= order['purchase_price'] * 1.01:
    #     next_value = 0.94

    # if exchange_rate_usd >= order['purchase_price'] * 1.02:
    #     next_value = 1.0

    if exchange_rate_usd >= order['purchase_price'] * 1.025:
        next_value = ((exchange_rate_usd - order['purchase_price']) / order['purchase_price']) + 1 - 0.005

    return max(next_value, prev_value)
    # if exchange_rate_usd <= order['purchase_price'] * RAISE_STOPLOSS_THRESHOLD:
    #     return order['stop_loss_percent']

    # adjusted = (((exchange_rate_usd - order['purchase_price']) / order['purchase_price'] + 1) * RAISE_STOPLOSS_VALUE)
    # return max(order['stop_loss_percent'], adjusted)

def main(csv_filename):
    while True:
        orders = load_orders_from_csv(csv_filename)
        print(datetime.datetime.now().strftime("%H:%M:%S"))

        for order in orders:
            if order['status'] == 'OPEN':
                try:
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
                    else:
                        order['stop_loss_percent'] = adjust_stoploss(order, exchange_rate_usd)
                except Exception as e:
                    print("An error occurred:", e)
            elif order['status'] == 'TRIGGERED':
                print(f"{order['coinbase_product_id']}\t TRIGGERED")

        write_orders_to_csv(orders, csv_filename)
        time.sleep(TIME_SLEEP_SECONDS)

if __name__ == "__main__":
    csv_filename = sys.argv[1]
    main(csv_filename)
