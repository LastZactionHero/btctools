import pandas as pd

# Load your data
file_path = './data/crypto_exchange_delta_dates.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Apply a moving average. Choose a window size that makes sense for your data
window_size = 5  # This is an example; you might need to adjust this based on your data characteristics

# Apply the moving average for each cryptocurrency column
for column in data.columns[3:]:  # Assuming cryptocurrency data starts from the 4th column
    data[column] = data[column].rolling(window=window_size).mean()

# Drop NaN values that result from the rolling mean
data = data.dropna()

data.to_csv("./data/crypto_exchange_delta_smooth.csv", index=False)