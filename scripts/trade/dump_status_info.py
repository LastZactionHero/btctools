import json
from sqlalchemy.orm import sessionmaker
from scripts.db.models import Order
import requests
from prettytable import PrettyTable
import time
import os
import datetime
from termcolor import colored

class DumpStatusInfo:
    def __init__(self, filename_json, filename_html):
        self.filename_html = filename_html
        self.filename_json = filename_json

    def save_status_info(self, timesource, last_buy_timestamp, last_coingecko_timestamp, usdc_available, prices, holdings, context):
        Session = sessionmaker(bind=context['engine'])
        session = Session()
        orders = session.query(Order).all()
        orders_grouped = {
            'open': [order.to_dict() for order in orders if order.status == 'OPEN'],
            'sold': [order.to_dict() for order in orders if order.status == 'SOLD']
        }
        session.close()

        # Create a dictionary with the required data
        status_info = {
            'timestamp': timesource.now(),
            'last_buy_timestamp': last_buy_timestamp,
            'last_coingecko_timestamp': last_coingecko_timestamp,
            'usdc_available': usdc_available,
            'holdings': holdings,
            'prices': prices.to_dict(),
            'orders': orders_grouped,
            'context': {k: v for k, v in context.items() if k != 'engine'}
        }

        self.dumpHTML(status_info)
        self.dumpJSON(status_info)


    def dumpJSON(self, status_info):
        status_info_json = json.dumps(status_info, indent=4)
        with open(self.filename_json, 'w') as f:
            f.write(status_info_json)

    def dumpHTML(self, status_info):
        usdc_available = status_info['usdc_available']
        timestamp = status_info['timestamp']
        holdings = status_info['holdings']
        prices = status_info['prices']
        orders = status_info['orders']

        with open(self.filename_html, 'w') as f:
            f.write('<html>\n')
            f.write('<head>\n')
            f.write('<style>\n')
            f.write('table {border-collapse: collapse;}\n')
            f.write('th, td {border: 1px solid black; padding: 8px;}\n')
            f.write('</style>\n')
            f.write('</head>\n')
            f.write('<body>\n')

            # Add summary table
            f.write('<h2>Summary</h2>\n')
            f.write('<table>\n')
            f.write('<tr><th>Metric</th><th>Value</th></tr>\n')
            f.write(f'<tr><td>USDC Available</td><td>${usdc_available:.2f}</td></tr>\n')
            f.write(f'<tr><td>Last Update</td><td>{datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")}</td></tr>\n')
            f.write('</table>\n')

            # Add holdings table
            f.write('<h2>Holdings</h2>\n')
            f.write('<table>\n')
            f.write('<tr><th>Currency</th><th>Quantity</th><th>Value in USDC</th></tr>\n')
            for currency, quantity in holdings.items():
                pair = f'{currency}-USDC'
                if pair in prices['bids']:
                    value_in_usdc = quantity * prices['bids'][pair]
                    f.write(f'<tr><td>{currency}</td><td>{quantity}</td><td>${value_in_usdc:.2f}</td></tr>\n')
            f.write('</table>\n')

            # Add sold orders table
            f.write('<h2>Sold Orders</h2>\n')
            f.write('<table>\n')
            f.write('<tr><th>ID</th><th>Action</th><th>Product ID</th><th>Quantity</th><th>Status</th><th>Sold At</th></tr>\n')
            for order in orders['sold']:
                f.write(f'<tr><td>{order["id"]}</td><td>{order["action"]}</td><td>{order["coinbase_product_id"]}</td><td>{order["quantity"]}</td><td>{order["status"]}</td><td>{order["sold_at"]}</td></tr>\n')
            f.write('</table>\n')

            # Add open orders table
            f.write('<h2>Open Orders</h2>\n')
            f.write('<table>\n')
            f.write('<tr><th>ID</th><th>Action</th><th>Symbol</th><th>Quantity</th><th>Purchase Price</th><th>Current Bid</th><th>Net</th><th>Net With Fees</th><th>Created At</th></tr>\n')
            for order in orders['open']:
                symbol = order['coinbase_product_id'].split('-')[0]
                current_price = prices['bids'].get(order['coinbase_product_id'], 0)
                net = (current_price - order['purchase_price']) * order['quantity']
                net_color = 'green' if net >= 0 else 'red'
                purchase_price = order['purchase_price']
                total_purchase_price = purchase_price * order['quantity'] * 1.01
                projected_sale_price = current_price * order['quantity'] * 0.99
                net_with_fees = projected_sale_price - total_purchase_price
                net_with_fees_color = 'green' if net_with_fees >= 0 else 'red'
                f.write(f'<tr><td>{order["id"]}</td><td>{order["action"]}</td><td>{symbol}</td><td>{order["quantity"]}</td><td>{round(purchase_price, 5)}</td><td>{round(current_price, 5)}</td><td style="color:{net_color}">{round(net, 2)}</td><td style="color:{net_with_fees_color}">{round(net_with_fees, 2)}</td><td>{order["created_at"]}</td></tr>\n')
            f.write('</table>\n')

            f.write('</body>\n')
            f.write('</html>\n')
