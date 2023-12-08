import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
file_path = './data/crypto_exchange_delta_smooth.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare sequences
def create_sequences(data, sequence_length):
    sequences = []
    output = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        output.append(data[i + sequence_length])
    return np.array(sequences), np.array(output)

sequence_length = 25  # Number of rows in each sequence
X, y = create_sequences(scaled_data, sequence_length)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=y.shape[1]))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=32)

# Making predictions
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)
predicted_output = model.predict(last_sequence)

# Inverse transform to get actual values
predicted_output = scaler.inverse_transform(predicted_output)
# print(predicted_output)

max_prediction = None
max_prediction_col = None

print(predicted_output)

# Assuming 'predicted_output' is your prediction array and 'data' is your DataFrame
predicted_values = predicted_output[0][3:]  # Assuming you want to process the first row of predictions
column_names = data.columns[3:]  # Skip the first column if it's 'timestamp'

# Associate each prediction with its corresponding column name
predictions_with_columns = list(zip(column_names, predicted_values))

# Sort the predictions in descending order
sorted_predictions = sorted(predictions_with_columns, key=lambda x: x[1], reverse=True)

# Print the sorted values
for column, value in sorted_predictions:
    print(f"{value}\t{column}")
