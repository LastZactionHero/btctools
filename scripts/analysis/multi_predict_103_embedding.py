import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

BAD_ROW_ANNOTATION = 999999
MAX_TIMESTAMP_DELTA = 300
SEQUENCE_MODULO = 10
EPOCHS = 1000

# Load CSV
file_path = sys.argv[1]
data = pd.read_csv(file_path)

data_price = data.drop('timestamp', axis=1).values

# Convert to delta
data_delta = (data_price[1:] - data_price[:-1]) / data_price[:-1]
data_price = data_price[1:]

data_price_annotated = np.copy(data_price)
data_delta_annotated = np.copy(data_delta)

data_price_filtered = np.copy(data_price)
data_delta_filtered = np.copy(data_delta)

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

for bad_row in bad_rows:
    if bad_row > 2:
        data_price_annotated[bad_row, :] = BAD_ROW_ANNOTATION
        np.delete(data_price_filtered, bad_row)

        data_delta_annotated[bad_row, :] = BAD_ROW_ANNOTATION
        np.delete(data_delta_filtered, bad_row)

# Correlation matrix vs bitcoin and etherium
data.set_index('timestamp', inplace=True)
correlation_matrix = data.corr()

# Scaling
delta_scaler = RobustScaler(with_centering=True, with_scaling=True)
data_delta_filtered_scaled = delta_scaler.fit_transform(data_delta_filtered)
data_delta_annotated_scaled = delta_scaler.transform(data_delta_annotated)

price_scaler = MinMaxScaler(feature_range=(0, 1))
data_price_filtered_scaled = price_scaler.fit_transform(np.concatenate((np.zeros((1, 253)), data_price_filtered), axis=0))
data_price_filtered_scaled = data_price_filtered_scaled[1:]
data_price_annotated_scaled = price_scaler.transform(np.concatenate((np.zeros((1, 253)), data_price_annotated), axis=0))
data_price_annotated_scaled = data_price_annotated_scaled[1:]

# Create Sequences
# - Last 5 days, in hour intervals
sequences_delta_last_5_days_hour_by_hour = []

MAX_LOOKBEHIND_MINUTES = 5 * 24 * 60
MAX_LOOKAHEAD_MINUTES = 0

for idx in range(MAX_LOOKBEHIND_MINUTES, (len(data_delta_annotated_scaled) - MAX_LOOKAHEAD_MINUTES) - 1):
    # Filter bad sequences
    for bad_row in bad_rows:
        if idx < (bad_row + MAX_LOOKBEHIND_MINUTES + 1) and idx > (bad_row - MAX_LOOKAHEAD_MINUTES - 1):
            continue

    if idx % 100 == 0:
        print(f"Sequence: {idx}/{len(data_delta_annotated_scaled)}")
        
    if idx % SEQUENCE_MODULO != 0:
        continue

    last_5_days_by_hour = list(range(idx - MAX_LOOKBEHIND_MINUTES, idx, 60))
    last_5_days_by_hour.append(idx)
    sequence_delta_last_5_days_by_hour = data_delta_annotated_scaled[last_5_days_by_hour]

    sequences_delta_last_5_days_hour_by_hour.append(sequence_delta_last_5_days_by_hour)


input_shape = sequences_delta_last_5_days_hour_by_hour[0].shape
embedding_size = 40  # Size of the embedding vector

inputs = Input(shape=input_shape)

# x = Flatten()(inputs)  # Flatten the input
# x = Dense(embedding_size, activation='relu')(x)
# x = Dropout(0.1)(x)
# encoder_output = Dense(embedding_size, activation='relu')(x)
x = LSTM(100, activation='tanh', return_sequences=True)(inputs)
x = Dropout(0.2)(x)
encoder_output = LSTM(embedding_size, activation='tanh')(x)  # Final encoder output
encoder_model = Model(inputs=inputs, outputs=encoder_output)

# x = Dense(input_shape[0] * input_shape[1], activation='linear')(encoder_output)
# outputs = Reshape(input_shape)(x)

x = LSTM(100, activation='tanh', return_sequences=True)(x)
outputs = TimeDistributed(Dense(input_shape[1]))(x)

# Create the full model
model = Model(inputs=inputs, outputs=outputs)
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Added gradient clipping
model.compile(optimizer=optimizer, loss='mse')
# model.compile(optimizer='adam', loss='mse')

sequences = np.array(sequences_delta_last_5_days_hour_by_hour)
sequences_train, sequences_val = train_test_split(sequences, test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = model.fit(sequences_train, sequences_train, epochs=EPOCHS, batch_size=32, validation_data=(sequences_val, sequences_val), callbacks=[early_stopping])

encoder_model = Model(inputs=inputs, outputs=encoder_output)
encoder_model.save('./models/multi_predict_103_encoder.h5')

real = sequences[-2:-1]
predicted = model.predict(real)

import pdb; pdb.set_trace()

