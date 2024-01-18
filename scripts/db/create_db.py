from models import Base, engine

# This script’s main purpose is to create the database, not define the models
Base.metadata.create_all(engine) 
