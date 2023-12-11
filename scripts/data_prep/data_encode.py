# Import necessary libraries
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from scaling import scale_rows

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(sys.argv[1], index_col=0)

# Define the window size
window_size = 30

# Create sliding windows for each coin
windows = {}
for coin in df.columns:
    if coin == "timestamp":
        next

    coin_data = df[coin].values
    # for i in range(len(coin_data) - window_size + 1):
    #     window = coin_data[i: i + window_size]
    #     scale_rows([window])[0] * 10
    #     windows.append(window)

# Convert the windows to a NumPy array
windows = np.array(windows)

import pdb; pdb.set_trace()