from scripts.db.models import Order, sessionmaker

class StatusDumper:
    def __init__(self, context):
        self.context = context

    def status_dump(self, timestamp, last_buy_timestamp):
        Session = sessionmaker(bind=self.context['engine'])
        session = Session()
        orders = session.query(Order).all
        session.close()
    
        return {
            "timestamp": timestamp,
            "last_buy_at": last_buy_timestamp,
            "orders": orders
        }