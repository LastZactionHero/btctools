from datetime import datetime

class Timesource:
    def __init__(self):
        pass

    def now(self):
        return int(datetime.now().timestamp())