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
DAY_SEQUENCE_LENGTH = 24 * 60
PREVIOUS_DAYS_LENGTH = 5
# PREDICTION_MINUTES = 24 * 60 #120
PREDICTION_MINUTES = 120
SEQUENCE_MODULO = 100

# Load data
data = pd.read_csv(sys.argv[1])
prediction_csv_file = sys.argv[2]

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
    sequences = []
    hourly_sequences = []
    output = []

    for i in range(len(data) - PREDICTION_MINUTES):
        for bad_row in bad_rows:
            if i < (bad_row + PREVIOUS_DAYS_LENGTH * 24 * 60 + 1) and i > (bad_row - PREDICTION_MINUTES - 1):
                continue

        if i <= PREVIOUS_DAYS_LENGTH * 24 * 60:
            continue
        if i % SEQUENCE_MODULO != 0:
            continue

        # Last day sequence
        sequences.append(delta[i - DAY_SEQUENCE_LENGTH:i])

        # Last week sequence
        index_array = np.flip((i - np.arange(0, PREVIOUS_DAYS_LENGTH * 24 * 60, 60)))
        hourly_sequences.append(price[index_array])

        # price_next_day_by_minute = price[i:i + PREDICTION_MINUTES]
        # next_day_max_prices = np.max(price_next_day_by_minute, axis=0)
        # next_day_min_prices = np.min(price_next_day_by_minute, axis=0)

        output.append(np.concatenate((
            # next_day_min_prices,
            # next_day_max_prices
            price[i + 60],
            price[i + 120]
        )))
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
optimizer = Adam(clipnorm=1.0)  # You can adjust the clipnorm value
model.compile(optimizer=optimizer, loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
model.fit([X_one_day, X_hours], y, epochs=100, batch_size=32, validation_data=(
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

# import numpy as np
# import pandas as pd
# import sys
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Flatten, Reshape, Dropout, concatenate
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # NEXT STEPS
# # - Build/Train the model
# #   INPUT:
# #   - dense1: last 5 days embedding
# #   - last1: last day delta @ 5 minutes
# #   - lstm2: last day price @ 5 minutes
# #   - similarities: btc/etc coeffecient
# #   OUTPUT:
# #   - next day min/max

# BAD_ROW_ANNOTATION = 999999
# MAX_TIMESTAMP_DELTA = 300
# SEQUENCE_MODULO = 300

# # Load CSV
# file_path = sys.argv[1]
# data = pd.read_csv(file_path)

# data_price = data.drop('timestamp', axis=1).values

# # Convert to delta
# data_delta = (data_price[1:] - data_price[:-1]) / data_price[:-1]
# data_price = data_price[1:]

# data_price_annotated = np.copy(data_price)
# data_delta_annotated = np.copy(data_delta)

# data_price_filtered = np.copy(data_price)
# data_delta_filtered = np.copy(data_delta)

# # Annotate and filter bad rows
# timestamps = data['timestamp']
# bad_rows = []
# for row_idx, timestamp in enumerate(timestamps):
#     if row_idx < len(timestamps) - 2:
#         next_timestamp = timestamps[row_idx + 1]
#         delta_timestamp = next_timestamp - timestamp
#         if delta_timestamp > MAX_TIMESTAMP_DELTA:
#             bad_rows.append(row_idx)
# bad_rows.sort(reverse=True)
# print(f"Bad Rows: {bad_rows}")

# for bad_row in bad_rows:
#     if bad_row > 2:
#         data_price_annotated[bad_row, :] = BAD_ROW_ANNOTATION
#         np.delete(data_price_filtered, bad_row)

#         data_delta_annotated[bad_row, :] = BAD_ROW_ANNOTATION
#         np.delete(data_delta_filtered, bad_row)

# # Correlation matrix vs bitcoin and etherium
# data.set_index('timestamp', inplace=True)
# correlation_matrix = data.corr()

# # Scaling
# delta_scaler = RobustScaler(with_centering=True, with_scaling=True)
# data_delta_filtered_scaled = delta_scaler.fit_transform(data_delta_filtered)
# data_delta_annotated_scaled = delta_scaler.transform(data_delta_annotated)

# price_scaler = MinMaxScaler(feature_range=(0, 1))
# data_price_filtered_scaled = price_scaler.fit_transform(data_price_filtered, axis=0)
# data_price_annotated_scaled = price_scaler.transform(data_price_annotated, axis=0)
# data_price_annotated_scaled = data_price_annotated_scaled[1:]
# # data_price_filtered_scaled = price_scaler.fit_transform(np.concatenate((np.zeros((1, 253)), data_price_filtered), axis=0))
# # data_price_filtered_scaled = data_price_filtered_scaled[1:]
# # data_price_annotated_scaled = price_scaler.transform(np.concatenate((np.zeros((1, 253)), data_price_annotated), axis=0))
# # data_price_annotated_scaled = data_price_annotated_scaled[1:]

# # Create Sequences
# # - Last 5 days, in hour intervals
# # - Last day, 5 minute intervals
# # sequences_delta_last_5_days_hour_by_hour = []
# # sequences_delta_last_day_by_minute = []
# sequences_price_last_day_by_minute = []
# sequences_next_days_extremes = []

# MAX_LOOKBEHIND_MINUTES = 5 * 24 * 60
# MAX_LOOKAHEAD_MINUTES = 24 * 60

# encoder = load_model("./models/multi_predict_103_encoder.h5")
# for idx in range(MAX_LOOKBEHIND_MINUTES, (len(data_delta_annotated_scaled) - MAX_LOOKAHEAD_MINUTES) - 1):
#     # Filter bad sequences
#     for bad_row in bad_rows:
#         if idx < (bad_row + MAX_LOOKBEHIND_MINUTES + 1) and idx > (bad_row - MAX_LOOKAHEAD_MINUTES - 1):
#             continue

#     if idx % 1000 == 0:
#         print(f"Sequence: {idx}/{len(data_delta_annotated_scaled)}")
        
#     if idx % SEQUENCE_MODULO != 0:
#         continue

#     # last_5_days_by_hour = list(range(idx - MAX_LOOKBEHIND_MINUTES, idx, 60))
#     # last_5_days_by_hour.append(idx)
#     # sequence_delta_last_5_days_by_hour = data_delta_annotated_scaled[last_5_days_by_hour]

#     last_day_by_5_minutes = list(range(idx - MAX_LOOKAHEAD_MINUTES, idx, 5))
#     last_day_by_5_minutes.append(idx)
#     # sequence_delta_last_day_by_minute = data_delta_annotated_scaled[last_day_by_5_minutes]
#     sequence_price_last_day_by_minute = np.transpose(data_price_annotated_scaled[last_day_by_5_minutes])
#     for s in sequence_price_last_day_by_minute:
#         sequences_price_last_day_by_minute.append(s)

#     # Max/Max Next Day Prices
#     price_next_day_by_minute = data_price_annotated_scaled[idx:idx+MAX_LOOKAHEAD_MINUTES]
#     next_day_max_prices = np.max(price_next_day_by_minute, axis=0)
    
#     for p in next_day_max_prices:
#         sequences_next_days_extremes.append(p)

#     # next_day_min_prices = np.min(price_next_day_by_minute, axis=0).reshape(1, -1)
#     # next_day_extreme_prices = np.concatenate((next_day_min_prices, next_day_max_prices), axis=0)
#     # next_day_extreme_prices = next_day_max_prices

#     # sequences_delta_last_5_days_hour_by_hour.append(sequence_delta_last_5_days_by_hour)
#     # sequences_delta_last_day_by_minute.append(sequence_delta_last_day_by_minute)
#     # sequences_price_last_day_by_minute.append(sequence_price_last_day_by_minute)
#     # sequences_next_days_extremes.append(next_day_extreme_prices)
        

# # Split hour-level data into training and testing sets
# X_prices_train, X_prices_test, y_train, y_test = train_test_split(
#     sequences_price_last_day_by_minute, sequences_next_days_extremes, test_size=0.2, random_state=42)

# # Define the first LSTM model for minute-level data
# input_prices = Input(shape=(1, len(X_prices_train[0])))  # Shape (1, 289) for a single time step
# prices_lstm = LSTM(units=100, return_sequences=True)(input_prices)
# prices_dropout = Dropout(0.2)(prices_lstm)
# prices_lstm_2 = LSTM(units=100, return_sequences=False)(prices_dropout)
# prices_output = Dropout(0.2)(prices_lstm_2)

# # # Add a Dense layer for final prediction
# dense_layer = Dense(units=1)(prices_output)

# # # Create the model
# model = Model(inputs=[input_prices], outputs=dense_layer)

# # # Compile the model
# optimizer = Adam(clipnorm=1.0)  # You can adjust the clipnorm value
# model.compile(optimizer=optimizer, loss='mean_squared_error')

# early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# # Train the model

# X_prices_train = np.array(X_prices_train).reshape(-1, 1, 289)
# X_prices_test = np.array(X_prices_test).reshape(-1, 1, 289)

# model.fit([np.array(X_prices_train)], np.array(y_train), epochs=20, batch_size=32, validation_data=(
#     [np.array(X_prices_test)], np.array(y_test)), callbacks=[early_stopping])
# # model.save(output_file)

# # latest_delta_sequence = delta_scaled[(-DAY_SEQUENCE_LENGTH-1):-1]

# # prediction = model.predict([np.array([X_one_day[-1]]), np.array([X_hours[-1]])])

# # unscaled = price_scaler.inverse_transform(prediction.reshape(2,253))
# # latest_unscaled = price_scaler.inverse_transform([price_scaled[-1]])

# # delta_unscaled = (unscaled - latest_unscaled) / latest_unscaled

# # df = pd.DataFrame(
# #     np.vstack([delta_unscaled[1], latest_unscaled[-1], unscaled[0], unscaled[1]]),
# #     columns=data.columns,
# #     index=['delta', 'latest', 'prediction_60', 'prediction_120']).round(6).T.sort_values('delta', ascending=False)
# # df.to_csv(prediction_csv_file, float_format='%.6f')