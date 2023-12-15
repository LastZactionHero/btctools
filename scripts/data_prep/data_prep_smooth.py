import pandas as pd
import sys

# Load your data
file_path = sys.argv[1]
data = pd.read_csv(file_path)

# Apply a moving average. Choose a window size that makes sense for your data
window_size = 5  # This is an example; you might need to adjust this based on your data characteristics

# Apply the moving average for each cryptocurrency column
for column in data.columns[3:]:  # Assuming cryptocurrency data starts from the 4th column
    data[column] = data[column].rolling(window=window_size).mean()

# Drop NaN values that result from the rolling mean
data = data.dropna()

data.to_csv(sys.argv[2], index=False)