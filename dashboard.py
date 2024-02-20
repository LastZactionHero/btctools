import requests
from prettytable import PrettyTable
import time
import os
import datetime
from termcolor import colored  # Uncomment for color coding


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
    timestamp = datetime.datetime.fromtimestamp(data["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
    summary_table.add_row(["Last Update", timestamp])
    print(summary_table)
    print("\n")

    # Holdings Table
    holdings_table = PrettyTable()
    holdings_table.field_names = ["Currency", "Quantity", "Value in USDC"]
    for currency, quantity in data["holdings"].items():
        # Assuming you can calculate the value in USDC for each holding
        pair = currency + "-USDC"
        if pair in data["prices"]["bids"]:
            value_in_usdc = (
                quantity * data["prices"]["bids"][pair]
            )  # Placeholder for actual calculation
            holdings_table.add_row([currency, quantity, f"${value_in_usdc:.2f}"])
    print(holdings_table)
    print("\n")

    # Sold Orders Table
    sold_orders_table = PrettyTable()
    sold_orders_table.field_names = [
        "ID",
        "Action",
        "Product ID",
        "Quantity",
        "Status",
        "Sold At",
    ]
    sold_orders = sorted(
        data["orders"]["sold"], key=lambda x: x["sold_at"], reverse=True
    )
    for order in sold_orders:
        sold_orders_table.add_row(
            [
                order["id"],
                order["action"],
                order["coinbase_product_id"],
                order["quantity"],
                order["status"],
                order["sold_at"],
            ]
        )
    print("SOLD")
    print(sold_orders_table)
    print("\n")

    # Open Orders Table
    open_orders_table = PrettyTable()
    open_orders_table.field_names = [
        "ID",
        "Action",
        "Symbol",
        "Quantity",
        "Purchase Price",
        "Current Bid",
        "Net",
        "Net With Fees",
        "Created At",
    ]
    open_orders = sorted(
        data["orders"]["open"], key=lambda x: x["created_at"], reverse=True
    )
    for order in open_orders:
        symbol = order["coinbase_product_id"].split("-")[0]
        current_price = data["prices"]["bids"].get(order["coinbase_product_id"], 0)
        net = (current_price - order["purchase_price"]) * order["quantity"]
        net_color = "green" if net >= 0 else "red"  # Determine color based on net value

        purchase_price = order["purchase_price"]
        total_purchase_price = purchase_price * order["quantity"] * 1.01
        projected_sale_price = current_price * order["quantity"] * 0.99
        net_with_fees = projected_sale_price - total_purchase_price
        net_with_fees_color = (
            "green" if net_with_fees >= 0 else "red"
        )  # Determine color based on net value

        open_orders_table.add_row(
            [
                order["id"],
                order["action"],
                symbol,
                order["quantity"],
                round(purchase_price, 5),
                round(current_price, 5),
                colored(round(net, 2), net_color),
                colored(round(net_with_fees, 2), net_with_fees_color),
                order["created_at"],
            ]
        )  # Use colored function to apply color
    print("OPEN")
    print(open_orders_table)
    print("\n")


def main():
    url = "http://magpie.asciisnowman.yachts/status.json"

    while True:
        os.system("cls" if os.name == "nt" else "clear")  # Clear console
        data = fetch_tradebot_status(url)
        display_dashboard(data)
        time.sleep(60)  # Sleep for 60 seconds


if __name__ == "__main__":
    main()
