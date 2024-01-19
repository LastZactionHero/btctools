import os
from dotenv import load_dotenv
from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
from simulated_broker import SimulatedBroker

# Load environment variables
load_dotenv()
api_key_name = os.getenv("COINBASE_API_KEY_NAME")
private_key = os.getenv("COINBASE_PRIVATE_KEY")
client = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(
    api_key_name, private_key)

broker = SimulatedBroker(client)
broker.reset()

broker.buy("113", "USDC-USDC", 1.0, 10000)
broker.buy(client_order_id="111", product_id="BTC-USDC",
           limit_price=100.0, base_size=20.0)
broker.buy("111", "ETH-USDC", 100, 20)