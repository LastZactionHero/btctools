import sys
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, concatenate
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

BAD_ROW_ANNOTATION = 999999
MAX_TIMESTAMP_DELTA = 300
DAY_SEQUENCE_LENGTH = 5 * 24
PREVIOUS_DAYS_LENGTH = 5
PREDICTION_MINUTES = 24 * 60
SEQUENCE_MODULO = 50

# Load data
data = pd.read_csv(sys.argv[1])
# prediction_csv_file = sys.argv[2]

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

# Prepare Delta Data Set
data = data.drop('timestamp', axis=1)
delta = (data[1:].values - data[:-1].values) / data[:-1].values
delta_scaler = RobustScaler(with_centering=True, with_scaling=True)
delta_scaled = delta_scaler.fit_transform(delta)

# Prepare Abs Price Data Set
price_scaler = MinMaxScaler(feature_range=(0, 1))
price_scaled = price_scaler.fit_transform(np.concatenate((np.zeros((1, 253)), data.values), axis=0))
price_scaled = price_scaled[2:]

# Split into Sequences
def create_sequences(delta, price):
    last_5_price_sequence = []
    last_5_delta_sequence = []
    output = []

    for i in range(len(data) - PREDICTION_MINUTES):
        for bad_row in bad_rows:
            if i < (bad_row + PREVIOUS_DAYS_LENGTH * 24 * 60 + 1) and i > (bad_row - PREDICTION_MINUTES - 1):
                continue

        if i <= PREVIOUS_DAYS_LENGTH * 24 * 60:
            continue
        if i % SEQUENCE_MODULO != 0:
            continue

        # Last week sequence
        index_array = np.flip((i - np.arange(0, PREVIOUS_DAYS_LENGTH * 24 * 60, 60)))
        last_5_delta_sequence.append(delta[index_array])
        last_5_price_sequence.append(price[index_array])

        prediction_index_array = i + 60 + np.arange(0, 24 * 60, 60)
        predictions = np.mean(delta[prediction_index_array], axis=1)
        output.append(predictions)

    return np.array(last_5_price_sequence), np.array(last_5_delta_sequence), np.array(output)

X_price, X_delta, y = create_sequences(delta_scaled, price_scaled)

# Split hour-level data into training and testing sets
X_price_train, X_price_test, y_train, y_test = train_test_split(
    X_price, y, test_size=0.2, random_state=42)
X_delta_train, X_delta_test, y_hour_train, y_hour_test = train_test_split(
    X_delta, y, test_size=0.2, random_state=42)

# Define the first LSTM model for minute-level data
input_minute = Input(shape=(X_price.shape[1], X_price.shape[2]))
minute_lstm = LSTM(units=1024, return_sequences=True)(input_minute)
minute_dropout = Dropout(0.2)(minute_lstm)
minute_lstm_2 = LSTM(units=500, return_sequences=False)(minute_dropout)
minute_output = Dropout(0.2)(minute_lstm_2)

# Define the second LSTM model for hour-level data
input_hour = Input(shape=(X_delta.shape[1], X_delta.shape[2]))
hour_lstm = LSTM(units=1024, return_sequences=True)(input_hour)
hour_dropout = Dropout(0.2)(hour_lstm)
hour_lstm_2 = LSTM(units=500, return_sequences=False)(hour_dropout)
hour_output = Dropout(0.2)(hour_lstm_2)

# # Combine the outputs of both models
combined = concatenate([minute_output, hour_output])

# # Add a Dense layer for final prediction
dense_layer = Dense(units=y.shape[1])(combined)

# # Create the model
model = Model(inputs=[input_minute, input_hour], outputs=dense_layer)

# # Compile the model
optimizer = Adam(clipnorm=1.0)  # You can adjust the clipnorm value
model.compile(optimizer=optimizer, loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
model.fit([X_price, X_delta], y, epochs=100, batch_size=32, validation_data=(
    [X_price_test, X_delta_test], y_test), callbacks=[early_stopping])

index_array = np.flip((len(delta_scaled) - 1 - np.arange(0, PREVIOUS_DAYS_LENGTH * 24 * 60, 60)))

latest_delta_sequence = delta_scaled[index_array]
latest_price_sequence = price_scaled[index_array]

next_day_prediction = model.predict([np.array([latest_price_sequence]), np.array([latest_delta_sequence])])[0]

for idx, prediction in enumerate(next_day_prediction):
    print(f"hour+{idx}:\t{prediction:2.4f}")



