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

orders_to_update = session.query(Order).filter(Order.id.in_([140,141,142,143,144,149])).all()

# Update the stop_loss_percent for each fetched order
for order in orders_to_update:
    order.stop_loss_percent = 0.782
    order.recovery_mode = True

session.commit()

# Close the session
session.close()