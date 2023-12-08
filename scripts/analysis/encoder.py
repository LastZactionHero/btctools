# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

def scale_window(arr):
    scaler = StandardScaler()
    scaled_data = []

    for row in arr:
        scaled_row = scaler.fit_transform(row.reshape(-1, 1)).flatten()
        scaled_data.append(scaled_row)

    return np.array(scaled_data)


# Load the CSV data into a pandas DataFrame
df = pd.read_csv('./data/crypto_exchange_rates.csv', index_col=0)

# Define the window size
window_size = 15

# Create sliding windows for each coin
windows = []
for coin in df.columns:
    if coin == "timestamp":
        next

    coin_data = scale_window([df[coin].values])[0]
    for i in range(len(coin_data) - window_size + 1):
        window = coin_data[i: i + window_size]
        windows.append(window)

# Convert the windows to a NumPy array
windows = np.array(windows)

# Split the data into training and testing sets
split_index = int(windows.shape[0] * 0.8)
train_windows = windows[:split_index]
test_windows = windows[split_index:]

# Create the autoencoder model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(window_size, activation='linear'))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_windows, train_windows, epochs=10, batch_size=32)

# Create the encoder model
encoder = Sequential()
encoder.add(model.layers[0])
encoder.add(model.layers[1])

# Freeze the encoder's weights
for layer in encoder.layers:
    layer.trainable = False

# Evaluate the model on the test set
loss = model.evaluate(test_windows, test_windows)
print('Test Loss:', loss)

# Extract latent representations from new time series data
new_windows = windows[0:10]
latent_representations = encoder.predict(new_windows)

# Use the latent representations for other tasks, such as classification