import initpath
import time
import os
import datetime
import logging
import logging_setup
from dotenv import load_dotenv
from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
from db.models import engine, Order, sessionmaker  
from print_utils import portfolio_table
from simulated_broker import SimulatedBroker

# Constants
CSV_FIELDNAMES = ['STATUS','ACTION','COINBASE_PRODUCT_ID','QUANTITY','PURCHASE_PRICE','STOP_LOSS_PERCENT','PROFIT_PERCENT']
FLOAT_FIELDS = ['QUANTITY','PURCHASE_PRICE','STOP_LOSS_PERCENT','PROFIT_PERCENT']
DATE_FORMAT = "%Y%m%d%H%M%S"
TIME_SLEEP_SECONDS = 10
RAISE_STOPLOSS_THRESHOLD = 1.012
SELL_STOPLOSS_FLOOR = 0.005

logging_setup.init_logging("sell.log")

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

def load_orders_from_database():
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        orders = session.query(Order).filter_by(status="OPEN").all()  # Filter only OPEN orders
        session.close()
        return orders
    except Exception as e:
        logging.error("Failed to load orders from the database: %s", e)
        return []

def update_order_status(order_id, new_status):
    try:
        Session = sessionmaker(bind=engine)
        session = Session()

        order_to_update = session.get(Order, order_id) 
        if order_to_update:
            order_to_update.status = new_status
            session.commit()
        else:
            logging.warning("Order with ID %s not found for update.", order_id)
    except Exception as e:
        logging.error("Failed to update order status: %s", e)
    finally:
        session.close()

def update_order_stoploss(order, new_stoploss_value):
    try:
        Session = sessionmaker(bind=engine)
        session = Session()

        order_to_update = session.get(Order, order.id)  # Get the Order from the database

        if order_to_update:
            logging.info("Updating stop-loss Order with ID %s to %f.", order.id, new_stoploss_value)
            order_to_update.stop_loss_percent = new_stoploss_value   # Update the stop-loss
            session.commit()    # Commit the change to the database
        else:
            # Handle the case where the order is not found (log a warning)
            logging.warning("Order with ID %s not found for stop-loss update.", order.id)
    except Exception as e:
        logging.error("Failed to update order stop-loss: %s", e)
    finally:
        session.close()

def create_and_execute_sell_order(order, exchange_rate_usd, client_order_id):
    try:
        broker.sell(
            client_order_id=client_order_id,
            buy_order_id=order.order_id,
            product_id=order.coinbase_product_id,
            limit_price=exchange_rate_usd,
            base_size=order.quantity)
        # TODO: Error handling
        update_order_status(order.id, "SOLD")
        
        logging.info("Sell order executed successfully for order ID %s", order.id)
    except Exception as e:
        logging.error("An error occurred while creating and executing sell order for order ID %s: %s", order.id, e)


def build_client_order_id(order):
    symbol = order.coinbase_product_id.split("-")[0]
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(DATE_FORMAT)
    return f"{order.action}_{symbol}_{timestamp}"

def print_order_details(order, exchange_rate_usd, client_order_id):
    message = f"{order.coinbase_product_id}\t bid is ${exchange_rate_usd:08.4f}\tpurchase ${order.purchase_price:08.4f}\tdelta {delta_since_purchase(order, exchange_rate_usd):0.2f}%\tlow ${stop_loss_price(order):08.4f}\thigh ${profit_price(order):08.4f}"
    logging.info(message)

def stop_loss_price(order):
    return order.purchase_price * order.stop_loss_percent

def profit_price(order):
    return order.purchase_price * order.profit_percent

def delta_since_purchase(order, exchange_rate_usd):
    return 100 * (exchange_rate_usd - order.purchase_price) / order.purchase_price

def should_trigger_order(order, exchange_rate_usd):
    return exchange_rate_usd >= profit_price(order) or exchange_rate_usd <= stop_loss_price(order)

def adjust_stoploss(order, exchange_rate_usd):
    prev_value = order.stop_loss_percent
    next_value = order.stop_loss_percent

    if exchange_rate_usd >= order.purchase_price * RAISE_STOPLOSS_THRESHOLD:
        next_value = ((exchange_rate_usd - order.purchase_price) / order.purchase_price) + 1 - SELL_STOPLOSS_FLOOR

    return max(next_value, prev_value)

def main():
    iter = 0
    while True:
        try:
            orders = load_orders_from_database()
            product_ids = set(map(lambda o: o.coinbase_product_id, orders))
            if(len(product_ids) == 0):
                continue
            
            best_bids = broker.get_best_bids(product_ids)
            for order in orders:
                exchange_rate_usd = float(best_bids[order.coinbase_product_id])

                client_order_id = build_client_order_id(order)
                print_order_details(order, exchange_rate_usd, client_order_id)

                if should_trigger_order(order, exchange_rate_usd):
                    create_and_execute_sell_order(order, exchange_rate_usd, client_order_id)
                else:
                    new_stoploss_value = adjust_stoploss(order, exchange_rate_usd)
                    if new_stoploss_value != order.stop_loss_percent:
                        update_order_stoploss(order, new_stoploss_value)
            
            if iter % 10 == 0:
                logging.info(f"\n{portfolio_table(broker.portfolio())}")
                logging.info(f"\n{portfolio_table(broker.holdings_usdc())}")
            iter += 1

            time.sleep(TIME_SLEEP_SECONDS)
        except Exception as  e:
            print(f"something went wrong {e}")
            # logger.error("Something went wrong")
if __name__ == "__main__":
    main()
