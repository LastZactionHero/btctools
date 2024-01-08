import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# NEXT STEPS
# - Skipping bad rows in sequence setup
# - Known global min/max
# - Build/Train the model
#   INPUT:
#   - lstm1: last 5 days delta
#   - lastm2: last day delta
#   - lstm3: last day price
#   - dense1: known global min/max )
#   OUTPUT:
#   - next day min/max

BAD_ROW_ANNOTATION = 999999
MAX_TIMESTAMP_DELTA = 300

# Load CSV
file_path = sys.argv[1]
data = pd.read_csv(file_path)

data_price = data.drop('timestamp', axis=1).values

# Convert to delta
data_delta = (data_price[1:] - data_price[:-1]) / data_price[:-1]
data_price = data_price[1:]

data_price_annotated = np.copy(data_price)
data_delta_annotated = np.copy(data_delta)

data_price_filtered = np.copy(data_price)
data_delta_filtered = np.copy(data_delta)

# Annotate and filter bad rows
timestamps = data['timestamp']
bad_rows = []
for row_idx, timestamp in enumerate(timestamps):
    if row_idx < len(timestamps) - 2:
        next_timestamp = timestamps[row_idx + 1]
        delta_timestamp = next_timestamp - timestamp
        if delta_timestamp > MAX_TIMESTAMP_DELTA:
            bad_rows.append(row_idx)
bad_rows.sort(reverse=True)
print(f"Bad Rows: {bad_rows}")

for bad_row in bad_rows:
    if bad_row > 2:
        data_price_annotated[bad_row, :] = BAD_ROW_ANNOTATION
        np.delete(data_price_filtered, bad_row)

        data_delta_annotated[bad_row, :] = BAD_ROW_ANNOTATION
        np.delete(data_delta_filtered, bad_row)

# Correlation matrix vs bitcoin and etherium
data.set_index('timestamp', inplace=True)
correlation_matrix = data.corr()

# Scaling
delta_scaler = RobustScaler(with_centering=True, with_scaling=True)
data_delta_filtered_scaled = delta_scaler.fit_transform(data_delta_filtered)
data_delta_annotated_scaled = delta_scaler.transform(data_delta_annotated)

price_scaler = MinMaxScaler(feature_range=(0, 1))
data_price_filtered_scaled = price_scaler.fit_transform(np.concatenate((np.zeros((1, 253)), data_price_filtered), axis=0))
data_price_filtered_scaled = data_price_filtered_scaled[1:]
data_price_annotated_scaled = price_scaler.transform(np.concatenate((np.zeros((1, 253)), data_price_filtered), axis=0))
data_price_annotated_scaled = data_price_annotated_scaled[1:]

# Create Sequences
# - Last 5 days, in hour intervals
# - Last day, 5 minute intervals
sequences_delta_last_5_days_hour_by_hour = []
sequences_delta_last_day_by_minute = []
sequences_price_last_day_by_minute = []
sequences_next_days_extremes = []

MAX_SEQUENCE_LENGTH_MINUTES = 5 * 24 * 60
MAX_LOOKAHEAD_MINUTES = 24 * 60
for idx in range(MAX_SEQUENCE_LENGTH_MINUTES, (len(data_delta_annotated_scaled) - MAX_LOOKAHEAD_MINUTES) - 1):
    # TODO: Filter bad sequences

    if idx % 1000 == 0:
        print(f"Sequence: {idx}/{len(data_delta_annotated_scaled)}")
        
    if idx % 100 != 0:
        continue

    last_5_days_by_hour = list(range(idx - 5 * 24 * 60, idx, 60))
    last_5_days_by_hour.append(idx)
    sequence_delta_last_5_days_by_hour = data_delta_annotated_scaled[last_5_days_by_hour]

    last_day_by_5_minutes = list(range(idx - 24 * 60, idx, 5))
    last_day_by_5_minutes.append(idx)
    sequence_delta_last_day_by_minute = data_delta_annotated_scaled[last_day_by_5_minutes]
    sequence_price_last_day_by_minute = data_price_annotated_scaled[last_day_by_5_minutes]

    # TODO: Known Global Min/Max

    # Max/Max Next Day Prices
    price_next_day_by_minute = data_price_annotated_scaled[idx:idx+1440]
    max_prices = np.max(price_next_day_by_minute, axis=0).reshape(1, -1)
    min_prices = np.min(price_next_day_by_minute, axis=0).reshape(1, -1)
    combined_prices = np.concatenate((max_prices, min_prices), axis=0)

    sequences_delta_last_5_days_hour_by_hour.append(sequence_delta_last_5_days_by_hour)
    sequences_delta_last_day_by_minute.append(sequence_delta_last_day_by_minute)
    sequences_price_last_day_by_minute.append(sequence_price_last_day_by_minute)
    sequences_next_days_extremes.append(combined_prices)

import pdb; pdb.set_trace()

# Inputs:
# - Embedding
# - Coin global max
# - Coin global min
# - Coin last 5 days, 5 minute intervals
#
# Predict:
# - Coin 1 day min
# - Coin 1 day max

