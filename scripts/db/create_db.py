from models import Base, init_db_engine

# This script’s main purpose is to create the database, not define the models
engine = init_db_engine("./db/magpie.db")
Base.metadata.create_all(engine) 
