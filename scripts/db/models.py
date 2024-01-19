from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv 
import os 

load_dotenv()
db_filename = os.getenv("DATABASE_FILENAME")

engine = create_engine(f'sqlite:///{db_filename}', echo=False)  # Use relative path
Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True) 
    status = Column(String)
    action = Column(String)
    coinbase_product_id = Column(String)
    quantity = Column(Float)
    purchase_price = Column(Float)
    stop_loss_percent = Column(Float)
    profit_percent = Column(Float)
    created_at = Column(DateTime, default=datetime.now)

class SimulatedHolding(Base):
    __tablename__ = 'simulated_holding'

    id = Column(Integer, primary_key=True) 
    created_at = Column(DateTime, default=datetime.now)
    status = Column(String, default="CURRENT")
    order_id = Column(String)
    coinbase_product_id = Column(String)
    quantity = Column(Float)
    purchase_price = Column(Float)