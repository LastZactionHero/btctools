# Import necessary libraries
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from scaling import scale_rows
from tensorflow.keras.callbacks import EarlyStopping

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(sys.argv[1], index_col=0)

# Define the window size
window_size = 30

# Create sliding windows for each coin
windows = []
for coin in df.columns:
    if coin == "timestamp":
        next

    coin_data = df[coin].values
    for i in range(len(coin_data) - window_size + 1):
        window = coin_data[i: i + window_size]
        window = scale_rows([window])[0] * 10
        windows.append(window)

# Convert the windows to a NumPy array
windows = np.array(windows)

# Split the data into training and testing sets
split_index = int(windows.shape[0] * 0.8)
train_windows = windows[:split_index]
test_windows = windows[split_index:]

# Create the autoencoder model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size-1, 1)))
# model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(window_size-1, activation='linear'))

early_stopping = EarlyStopping(
    monitor='loss',  # Monitor the loss value
    min_delta=0.001,  # Minimum change in loss value to trigger stopping
    patience=5,  # Number of epochs with no improvement before stopping
)

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_windows, train_windows, epochs=10, batch_size=32, callbacks=[early_stopping])
model.save("./data/encoder.h5")

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

import pdb; pdb.set_trace()
# Extract latent representations from new time series data
new_windows = windows[0:10]
latent_representations = encoder.predict(new_windows)

# Use the latent representations for other tasks, such as classification