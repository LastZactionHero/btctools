import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
from datetime import datetime, timedelta
from scipy.spatial import KDTree
from scripts.trade.coingecko_coinbase_pairs import gecko_coinbase_currency_map


def remove_non_finite_values(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    return df

def find_largest_timestamp_gap_last_5_days(df):
    current_time = datetime.utcnow()
    five_days_ago = current_time - timedelta(days=5)
    
    df = df[df['timestamp'] >= five_days_ago.timestamp()]
    timestamps = df['timestamp'].astype(int).sort_values()
    
    largest_gap = 0
    start_of_gap = 0
    end_of_gap = 0
    
    for i in range(1, len(timestamps)):
        gap = timestamps.iloc[i] - timestamps.iloc[i-1]
        if gap > largest_gap:
            largest_gap = gap
            start_of_gap = timestamps.iloc[i-1]
            end_of_gap = timestamps.iloc[i]
    
    return start_of_gap, end_of_gap

csv_file_path = "./data/crypto_exchange_rates.csv"
df = remove_non_finite_values(csv_file_path)
start_of_gap, end_of_gap = find_largest_timestamp_gap_last_5_days(df)
print("Start of the gap in the last 5 days:", datetime.utcfromtimestamp(start_of_gap))
print("End of the gap in the last 5 days:", datetime.utcfromtimestamp(end_of_gap))

# Load the 'asks.csv' data
asks_df = pd.read_csv("./data/asks.csv")
asks_df['timestamp'] = pd.to_numeric(asks_df['timestamp'], errors='coerce')
asks_df.dropna(subset=['timestamp'], inplace=True)

# Generate new timestamps
new_timestamps = list(range(start_of_gap + 60, end_of_gap, 60))

# Prepare a KDTree for efficient nearest timestamp search
kd_tree = KDTree(asks_df[['timestamp']].values)

def find_nearest_ask_timestamp(target):
    distance, index = kd_tree.query([[target]])
    return asks_df.iloc[index]['timestamp']

# Create a new DataFrame to hold the interpolated data
new_rows = pd.DataFrame(new_timestamps, columns=['timestamp'])
for column_name in df.columns:
    if column_name != 'timestamp':  # Skip timestamp since it's already added
        # Initialize all other columns with 0's but set dtype to float64 to avoid the warning
        new_rows[column_name] = 0.0

# Populate the new DataFrame
for column_name in df.columns:
    if column_name == 'timestamp':
        continue  # Skip the timestamp column itself
    coinbase_name = gecko_coinbase_currency_map.get(column_name, None)
    
    # Initialize the column in new_rows with 0's
    new_rows[column_name] = 0
    
    if coinbase_name and coinbase_name in asks_df.columns:
        for i, timestamp in enumerate(new_rows['timestamp']):
            nearest_timestamp = find_nearest_ask_timestamp(timestamp)
            # Access the values in a way that ensures no FutureWarning about dtype
            values = asks_df[asks_df['timestamp'] == nearest_timestamp][coinbase_name].values
            if len(values) > 0:
                # Assign the value explicitly converting to float to ensure compatibility
                new_rows.at[i, column_name] = float(values[0])
            else:
                # Use 0.0 to explicitly indicate floating-point
                new_rows.at[i, column_name] = 0.0

# Here, you may want to merge new_rows back into your original DataFrame or handle it as needed
print(new_rows)