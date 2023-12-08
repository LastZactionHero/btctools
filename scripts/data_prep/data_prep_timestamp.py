import pandas as pd
from datetime import datetime, timezone

# Load your data
file_path = './data/crypto_exchange_delta.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Convert Unix timestamp to datetime object
# Assuming your timestamps are in seconds
data['datetime'] = pd.to_datetime(data['timestamp'], unit='s', utc=True)

# Extract day of the week (Monday=0, Sunday=6)
data['day_of_week'] = data['datetime'].dt.weekday

# Extract minute of the day (0 = midnight GMT)
data['minute_of_day'] = data['datetime'].dt.hour * 60 + data['datetime'].dt.minute

columns = ['timestamp', 'day_of_week', 'minute_of_day'] + [col for col in data.columns if col not in ['timestamp', 'day_of_week', 'minute_of_day']]
data = data[columns]
data = data.drop(columns=['datetime'])

data.to_csv("./data/crypto_exchange_delta_dates.csv", index=False)