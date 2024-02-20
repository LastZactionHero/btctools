import requests
from prettytable import PrettyTable
import time
# from termcolor import colored  # Uncomment for color coding

def fetch_tradebot_status(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def display_dashboard(data):
    if not data:
        print("No data to display.")
        return

    # Summary Section
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    summary_table.add_row(["USDC Available", f"${data['usdc_available']:.2f}"])
    summary_table.add_row(["Last Update", data['timestamp']])
    print(summary_table)

    # Holdings Table
    holdings_table = PrettyTable()
    holdings_table.field_names = ["Currency", "Quantity", "Value in USDC"]
    for currency, quantity in data['holdings'].items():
        # Assuming you can calculate the value in USDC for each holding
        value_in_usdc = quantity * 1  # Placeholder for actual calculation
        holdings_table.add_row([currency, quantity, f"${value_in_usdc:.2f}"])
    print(holdings_table)

    # Orders Table
    orders_table = PrettyTable()
    orders_table.field_names = ["ID", "Action", "Product ID", "Quantity", "Status"]
    for order in data['orders']['open'] + data['orders']['sold']:
        orders_table.add_row([order['id'], order['action'], order['coinbase_product_id'], order['quantity'], order['status']])
    print(orders_table)

    # Prices Table (Simplified view)
    prices_table = PrettyTable()
    prices_table.field_names = ["Pair", "Bid", "Ask"]
    for pair, bid in data['prices']['bids'].items():
        ask = data['prices']['asks'].get(pair)
        prices_table.add_row([pair, f"${bid}", f"${ask}"])
    print(prices_table)

    # Context Info
    context_table = PrettyTable()
    context_table.field_names = ["Setting", "Value"]
    for key, value in data['context'].items():
        context_table.add_row([key, value])
    print(context_table)

def main():
    url = "http://magpie.asciisnowman.yachts/status.json"
    
    while True:
        data = fetch_tradebot_status(url)
        display_dashboard(data)
        time.sleep(60)  # Sleep for 60 seconds

if __name__ == "__main__":
    main()