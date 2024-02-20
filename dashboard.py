import requests
from prettytable import PrettyTable
import time
import os
import datetime
from termcolor import colored
def fetch_tradebot_status(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def display_summary(data):
    summary_table = PrettyTable(["Metric", "Value"])
    summary_table.add_row(["USDC Available", f"${data['usdc_available']:.2f}"])
    timestamp = datetime.datetime.fromtimestamp(data["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
    summary_table.add_row(["Last Update", timestamp])
    print(summary_table)


def display_holdings(data):
    holdings_table = PrettyTable(["Currency", "Quantity", "Value in USDC"])
    for currency, quantity in data["holdings"].items():
        pair = currency + "-USDC"
        if pair in data["prices"]["bids"]:
            value_in_usdc = quantity * data["prices"]["bids"][pair]
            holdings_table.add_row([currency, quantity, f"${value_in_usdc:.2f}"])
    print(holdings_table)


def display_sold_orders(data):
    sold_orders_table = PrettyTable([
        "ID",
        "Action",
        "Product ID",
        "Quantity",
        "Status",
        "Sold At",
    ])
    sold_orders = sorted(data["orders"]["sold"], key=lambda x: x["sold_at"], reverse=True)
    for order in sold_orders:
        sold_orders_table.add_row([
            order["id"],
            order["action"],
            order["coinbase_product_id"],
            order["quantity"],
            order["status"],
            order["sold_at"],
        ])
    print("SOLD")
    print(sold_orders_table)


def display_open_orders(data):
    open_orders_table = PrettyTable([
        "ID",
        "Action",
        "Symbol",
        "Quantity",
        "Purchase Price",
        "Current Bid",
        "Net",
        "Net With Fees",
        "Created At",
    ])
    open_orders = sorted(data["orders"]["open"], key=lambda x: x["created_at"], reverse=True)
    for order in open_orders:
        symbol = order["coinbase_product_id"].split("-")[0]
        current_price = data["prices"]["bids"].get(order["coinbase_product_id"], 0)
        net = (current_price - order["purchase_price"]) * order["quantity"]
        net_color = "green" if net >= 0 else "red"

        purchase_price = order["purchase_price"]
        total_purchase_price = purchase_price * order["quantity"] * 1.01
        projected_sale_price = current_price * order["quantity"] * 0.99
        net_with_fees = projected_sale_price - total_purchase_price
        net_with_fees_color = "green" if net_with_fees >= 0 else "red"

        open_orders_table.add_row([
            order["id"],
            order["action"],
            symbol,
            order["quantity"],
            round(purchase_price, 5),
            round(current_price, 5),
            colored(round(net, 2), net_color),
            colored(round(net_with_fees, 2), net_with_fees_color),
            order["created_at"],
        ])
    print("OPEN")
    print(open_orders_table)


def display_dashboard(data):
    if not data:
        print("No data to display.")
        return

    display_summary(data)
    print("\n")
    display_holdings(data)
    print("\n")
    display_sold_orders(data)
    print("\n")
    display_open_orders(data)
    print("\n")


def main():
    url = "http://magpie.asciisnowman.yachts/status.json"

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        data = fetch_tradebot_status(url)
        display_dashboard(data)
        time.sleep(60)


if __name__ == "__main__":
    main()
