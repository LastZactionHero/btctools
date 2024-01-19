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
print(portfolio_table(broker.portfolio()))
