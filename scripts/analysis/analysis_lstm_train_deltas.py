import sys
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, concatenate
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

DAY_SEQUENCE_LENGTH = 24 * 60
PREVIOUS_DAYS_LENGTH = 5
PREDICTION_MINUTES = 120
SEQUENCE_MODULO = 30

# Load data
data = pd.read_csv(sys.argv[1])
prediction_csv_file = sys.argv[2]
output_file = sys.argv[3]

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
    sequences = []
    hourly_sequences = []
    output = []

    for i in range(len(data) - PREDICTION_MINUTES):
        if i <= PREVIOUS_DAYS_LENGTH * 24 * 60:
          continue
        if i % SEQUENCE_MODULO != 0:
          continue

        # Last day sequence
        sequences.append(delta[i - DAY_SEQUENCE_LENGTH:i])

        # Last week sequence
        index_array = np.flip((i - np.arange(0, PREVIOUS_DAYS_LENGTH * 24 * 60, 60)))
        hourly_sequences.append(price[index_array])

        output.append(np.concatenate((
            price[i + 60],
            price[i + 120])
        ))
    return np.array(sequences), np.array(hourly_sequences), np.array(output)

X_one_day, X_hours, y = create_sequences(delta_scaled, price_scaled)

# Split hour-level data into training and testing sets
X_one_day_train, X_one_day_test, y_train, y_test = train_test_split(
    X_one_day, y, test_size=0.2, random_state=42)
X_hour_train, X_hour_test, y_hour_train, y_hour_test = train_test_split(
    X_hours, y, test_size=0.2, random_state=42)

# Define the first LSTM model for minute-level data
input_minute = Input(shape=(X_one_day.shape[1], X_one_day.shape[2]))
minute_lstm = LSTM(units=1024, return_sequences=True)(input_minute)
minute_dropout = Dropout(0.2)(minute_lstm)
minute_lstm_2 = LSTM(units=500, return_sequences=False)(minute_dropout)
minute_output = Dropout(0.2)(minute_lstm_2)

# Define the second LSTM model for hour-level data
input_hour = Input(shape=(X_hours.shape[1], X_hours.shape[2]))
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
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
model.fit([X_one_day, X_hours], y, epochs=20, batch_size=32, validation_data=(
    [X_one_day_test, X_hour_test], y_test), callbacks=[early_stopping])
# model.save(output_file)

latest_delta_sequence = delta_scaled[(-DAY_SEQUENCE_LENGTH-1):-1]

prediction = model.predict([np.array([X_one_day[-1]]), np.array([X_hours[-1]])])

unscaled = price_scaler.inverse_transform(prediction.reshape(2,253))
latest_unscaled = price_scaler.inverse_transform([price_scaled[-1]])

delta_unscaled = (unscaled - latest_unscaled) / latest_unscaled

df = pd.DataFrame(
    np.vstack([delta_unscaled[1], latest_unscaled[-1], unscaled[0], unscaled[1]]),
    columns=data.columns,
    index=['delta', 'latest', 'prediction_60', 'prediction_120']).round(6).T.sort_values('delta', ascending=False)
df.to_csv(prediction_csv_file, float_format='%.6f')