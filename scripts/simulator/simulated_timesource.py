class SimulatedTimesource:
    def __init__(self, timestamp):
        self.timestamp = int(timestamp)

    def now(self):
        return self.timestamp
    
    def set(self, timestamp):
        self.timestamp = int(timestamp)