import argparse
import os
import shutil
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from scripts.db.models import Base, Order, init_db_engine

# Parse command line arguments
parser = argparse.ArgumentParser(description='Delete an Order by ID')
parser.add_argument('order_id', type=int, help='ID of the Order to delete')
args = parser.parse_args()

# Backup the existing database file
db_filename = "./db/live.db"
backup_filename = f"./db/live_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.db"
shutil.copy2(db_filename, backup_filename)

# Connect to the database
engine = init_db_engine(db_filename)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

try:
    # Find the Order by ID
    order = session.query(Order).filter(Order.id == args.order_id).one()

    # Delete the Order
    session.delete(order)
    session.commit()
    print(f"Order with ID {args.order_id} deleted successfully.")
except Exception as e:
    session.rollback()
    print(f"Error deleting Order with ID {args.order_id}: {str(e)}")

session.close()
