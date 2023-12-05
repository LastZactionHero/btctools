import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from a CSV file
def load_data_from_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp = int(row['timestamp'])
            price_usd = float(row['price_usd'])
            data.append((timestamp, price_usd))
    return data

# Create a time series dataset
def create_time_series_dataset(data, window_size=10):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][1])  # Price for the next time

    return np.array(X), np.array(y)

# Define the RNN model
def create_rnn_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    # Load data from a CSV file
    data = load_data_from_csv("bitcoin_data.csv")  # Replace with your CSV file path

    # Create a time series dataset
    window_size = 10
    X, y = create_time_series_dataset(data, window_size)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape input data to be 3D (batch_size, time_steps, input_dim)
    input_shape = (X_train.shape[1], X_train.shape[2])
    X_train = X_train.reshape((-1, *input_shape))
    X_test = X_test.reshape((-1, *input_shape))

    # Create and train the RNN model
    model = create_rnn_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")
