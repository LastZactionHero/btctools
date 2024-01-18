from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)  # Auto-incrementing primary key
    status = Column(String)
    action = Column(String)
    coinbase_product_id = Column(String)
    quantity = Column(Float)
    purchase_price = Column(Float)
    stop_loss_percent = Column(Float)
    profit_percent = Column(Float)
    created_at = Column(DateTime, default=datetime.now)  # Timestamp 