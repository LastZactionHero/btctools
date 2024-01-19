import os
from dotenv import load_dotenv
from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
from prettytable import PrettyTable
from simulated_broker import SimulatedBroker


def portfolio_table(portfolio):
    table = PrettyTable(["Currency", "Balance (Coin)",
                        "Balance (USD)", "Allocation"])
    table.align["Currency"] = "l"
    table.align["Balance (Coin)"] = "r"
    table.align["Balance (USD)"] = "r"
    table.align["Allocation"] = "r"
    for holding in portfolio:
        table.add_row([
            holding.currency,
            holding.balance_coin,
            "{:.2f}".format(holding.balance_usd),
            "{:.2f}%".format(holding.allocation)
        ])
    return table


# Load environment variables
load_dotenv()
api_key_name = os.getenv("COINBASE_API_KEY_NAME")
private_key = os.getenv("COINBASE_PRIVATE_KEY")
client = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(
    api_key_name, private_key)

broker = SimulatedBroker(client)
broker.reset()

bids = broker.get_best_bids(["BTC-USDC", "ETH-USDC"])
print(bids)

asks = broker.get_best_asks(["BTC-USDC", "ETH-USDC"])
print(asks)

print("Buying...")
broker.buy("113", "USDC-USDC", 1.0, 10000)
broker.buy(client_order_id="111", product_id="BTC-USDC",
           limit_price=100.0, base_size=20.0)
broker.buy("111", "ETH-USDC", 100, 20)
print(portfolio_table(broker.portfolio()))

print("Selling...")
broker.sell(client_order_id="112", product_id="BTC-USDC",
            limit_price=200.0, base_size=20.0)
print(portfolio_table(broker.portfolio()))

print("Reseting...")
broker.reset()
print(portfolio_table(broker.portfolio()))
