import sys
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, concatenate
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

TRAINING_SEQUENCE_LENGTH = 1440
HOUR_SEQUENCE_LENGTH = 120
PREDICTION_SEQUENCE_LENGTH = 120

# Load data
data = pd.read_csv(sys.argv[1])
output_file = sys.argv[3]
prediction_csv_file = sys.argv[2]

# TODO: 
# Minute-by-minute is delta based. Scaler is global over entire data set, or logarithmic to empahsises lower change and soften highs
# Hour-by-hour LSTM is 0-1 grounded to cover the range relative to the max

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.concatenate((np.zeros((1, 254)), data.values), axis=0))
scaled_data = scaled_data[1:]

output_intervals = [60, 120]

def create_sequences(data):
    days_back = 5

    sequences = []
    hourly_sequences = []
    output = []
    for i in range(len(data) - 120):
        if i <= days_back * 24 * 60:
          continue

        if i % 30 != 0:
          continue

        # Last day sequence
        sequences.append(data[i - TRAINING_SEQUENCE_LENGTH:i])

        # Last week sequence
        index_array = np.flip((i - np.arange(0, days_back * 24 * 60, 60)))
        hourly_sequences.append(data[index_array])


        # list(map(lambda x: data[i + x], output_intervals))
        output.append(np.concatenate((
            data[i + 60],
            data[i + 120])
        ))
    return np.array(sequences), np.array(hourly_sequences), np.array(output)

X_one_day, X_hours, y = create_sequences(scaled_data)

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
model.fit([X_one_day, X_hours], y, epochs=10, batch_size=32, validation_data=(
    [X_one_day_test, X_hour_test], y_test), callbacks=[early_stopping])
model.save(output_file)

latest_sequence = scaled_data[(-TRAINING_SEQUENCE_LENGTH-1):-1]
index_array = np.flip((0 - np.arange(0, 5 * 1440, 60)))
latest_hourly_sequence = scaled_data[index_array]

# predicted_sequence = leatest_sequence
prediction = model.predict([np.array([latest_sequence]), np.array([latest_hourly_sequence])])

unscaled = scaler.inverse_transform(prediction.reshape(2,254))
latest_unscaled = scaler.inverse_transform(latest_sequence)

delta_unscaled = (unscaled[-1] - latest_unscaled[-1]) / latest_unscaled[-1]

delta_named = []
for idx, delta in enumerate(delta_unscaled):
    delta_named = np.append(delta_named, {
        "coin": data.columns[idx],
        "delta": delta.round(2)
    })

delta_named = sorted(delta_named, key=lambda x: x['delta'], reverse=True)

for delta in delta_named:
    print("{}: {}".format(delta['coin'], delta['delta']))

df = pd.DataFrame(
    np.vstack([delta_unscaled.reshape(1, -1), latest_unscaled[-1], unscaled[0], unscaled[1]]),
    columns=data.columns,
    index=['delta', 'latest', 'prediction_60', 'prediction_120']).drop('timestamp', axis=1).round(6).T.sort_values('delta', ascending=False)
df.to_csv(prediction_csv_file, float_format='%.6f')