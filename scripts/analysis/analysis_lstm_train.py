import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

TRAINING_SEQUENCE_LENGTH = 1440
PREDICTION_SEQUENCE_LENGTH = 120

# Load data
file_path = sys.argv[1]
data = pd.read_csv(file_path)
output_file = sys.argv[2]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare sequences
def create_sequences(data, sequence_length):
    sequences = []
    output = []
    for i in range(len(data) - sequence_length):
        if i % 25 != 0:
            continue
        sequences.append(data[i:i + sequence_length])
        output.append(data[i + sequence_length])
    return np.array(sequences), np.array(output)

X, y = create_sequences(scaled_data, TRAINING_SEQUENCE_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential()
model.add(LSTM(units=1024, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=500, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=y.shape[1]))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
model.fit(X, y, epochs=500, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
model.save(output_file)

# loss, accuracy = model.evaluate(X_test, y_test)
# print("Accuracy: {:.2f}%".format(accuracy * 100))
# model = load_model(output_file)

# Making predictions
predictions = None
last_sequence = scaled_data[-TRAINING_SEQUENCE_LENGTH:]

for i in range(PREDICTION_SEQUENCE_LENGTH):
    print("Prediction {}/{}".format(i, PREDICTION_SEQUENCE_LENGTH))
    predicted_output = model.predict(np.array([last_sequence]))

    # import pdb; pdb.set_trace()
    if predictions is None:
        predictions = np.array(predicted_output)
    else:
        predictions = np.concatenate((predictions, predicted_output), axis=0)

    last_sequence = np.concatenate((last_sequence, predicted_output), axis=0)
    last_sequence = last_sequence[1:]

prediction = scaler.inverse_transform(predictions) / 100
prediction_totals = np.sum(predictions, axis=0)

coin_predictions = zip(data.columns, prediction_totals)
coin_predictions = sorted(coin_predictions, key=lambda x: x[1], reverse=True)

for coin, prediction in coin_predictions:
    print("{:.5f}:\t{}".format(prediction, coin))