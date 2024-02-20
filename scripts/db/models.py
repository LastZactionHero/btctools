from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv 
import os 

# load_dotenv()
# db_filename = os.getenv("DATABASE_FILENAME")

# engine = None
Base = declarative_base()

def init_db_engine(filename):
    return create_engine(f'sqlite:///{filename}', echo=False)  # Use relative path
    

class Order(Base):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True) 
    order_id = Column(String)
    status = Column(String)
    action = Column(String)
    coinbase_product_id = Column(String)
    quantity = Column(Float)
    purchase_price = Column(Float)
    purchase_time_spread_percent = Column(Float)
    stop_loss_percent = Column(Float)
    profit_percent = Column(Float)
    predicted_max_delta = Column(Float)
    predicted_min_delta = Column(Float)
    num_predictions_over_hit = Column(Integer)
    max_delta_average=Column(Float)
    max_delta_std=Column(Float)
    min_delta_average=Column(Float)
    min_delta_std=Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.now)
    sold_at = Column(DateTime(timezone=True))
    recovery_mode = Column(Boolean, default=False)
    recovery_set = Column(Boolean, default=False)

    live_ordered = Column(Boolean, default=False)
    live_sold = Column(Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'order_id': self.order_id,
            'status': self.status,
            'action': self.action,
            'coinbase_product_id': self.coinbase_product_id,
            'quantity': self.quantity,
            'purchase_price': self.purchase_price,
            'purchase_time_spread_percent': self.purchase_time_spread_percent,
            'stop_loss_percent': self.stop_loss_percent,
            'profit_percent': self.profit_percent,
            'predicted_max_delta': self.predicted_max_delta,
            'predicted_min_delta': self.predicted_min_delta,
            'num_predictions_over_hit': self.num_predictions_over_hit,
            'max_delta_average': self.max_delta_average,
            'max_delta_std': self.max_delta_std,
            'min_delta_average': self.min_delta_average,
            'min_delta_std': self.min_delta_std,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'sold_at': self.sold_at.strftime('%Y-%m-%d %H:%M:%S') if self.sold_at else None,
            'recovery_mode': self.recovery_mode,
            'recovery_set': self.recovery_set,
            'live_ordered': self.live_ordered,
            'live_sold': self.live_sold
        }

class SimulatedHolding(Base):
    __tablename__ = 'simulated_holding'

    id = Column(Integer, primary_key=True) 
    created_at = Column(DateTime, default=datetime.now)
    status = Column(String, default="CURRENT")
    order_id = Column(String)
    coinbase_product_id = Column(String)
    quantity = Column(Float)
    purchase_price = Column(Float)