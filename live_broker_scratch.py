import pandas as pd
import time
from sqlalchemy import and_
from dotenv import load_dotenv
from scripts.trade.buyer import Buyer
from scripts.trade.seller import Seller
from scripts.trade.buyer_prediction_model import BuyerPredictionModel
from scripts.live.broker import Broker
from scripts.live.timesource import Timesource
# from scripts.live.ful import Fullcoin
from scripts.db.models import init_db_engine, Base
from scripts.db.models import Order, sessionmaker
from datetime import datetime, timedelta
import logging

load_dotenv()

engine = init_db_engine("./db/live.db")
Session = sessionmaker(bind=engine)
session = Session()

# Assuming your Orders table is mapped to a class named 'Order'
session.query(Order).filter(Order.id.in_([13, 14, 15, 16, 17])).delete()
session.commit()  # Important: Commit the changes
session.close()
#engine = init_db_engine("./db/live.db")
#Session = sessionmaker(bind=engine)
#session = Session()

#orders_to_delete = session.query(Order).filter(Order.id.in_([13, 14, 15, 16, 17]))

# Delete the queried records
#session.delete(orders_to_delete)

# Commit the changes to the database
#session.commit()
#
#session.close()

#logger = logging.getLogger('LiveLogger')
#broker = Broker(logger, {'live_trades': True})

#print(broker.usdc_available())

#b#roker.buy("PREP_001", "GFI-USDC", 50, 1.2834)
# broker.sell("PREP_002", "BADGER-USDC", 1, 4.36)
# prices = broker.prices()
# ask = prices.ask("BTRST")
# ask = 0.915556923992
# x = broker.buy("PREP_007", "BTRST-USDC", 5.102022011, ask)
# # print(x)
# DB_FILENAME = "./db/live.db"
# engine = init_db_engine(DB_FILENAME)
# Session = sessionmaker(bind=engine)
# session = Session()

# def order_summary(order):
#     return {
#         "symbol": order.coinbase_product_id.split("-")[0],
#         "ordered_at": order.created_at,
#         "status": order.status
#     }
# new_buys = session.query(Order).filter(and_(Order.status == "OPEN", Order.created_at > datetime.now() - timedelta(days=1))).all()
# new_sells = session.query(Order).filter(and_(Order.status == "SOLD", Order.sold_at > datetime.now() - timedelta(days=1))).all()

# print(list(map(order_summary, new_buys)))
# print(list(map(order_summary, new_sells)))


# session.close()
