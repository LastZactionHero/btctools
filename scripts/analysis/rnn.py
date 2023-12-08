import argparse
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import time

# Load data from a CSV file
def load_data_from_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            price_usd = float(row['price_usd'])
            data.append(price_usd)
    return data

# Create a time series dataset
def create_time_series_dataset(data, window_size=10, prediction_steps=10):
    X, y = [], []

    for i in range(len(data) - window_size - prediction_steps + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+prediction_steps])

    return np.array(X), np.array(y)

# Define the RNN model
def create_rnn_model(input_shape, prediction_steps):
    model = keras.Sequential([
        keras.layers.LSTM(50, activation='relu', input_shape=input_shape),
        keras.layers.Dense(prediction_steps)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Use the trained model to make predictions for the next steps
def predict_next_steps(model, X, prediction_steps):
    # Reshape the input data to be 3D (batch_size, time_steps, input_dim)
    X = X.reshape((1, *X.shape))

    # Make predictions
    predicted_prices = model.predict(X)[0]
    return predicted_prices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next 10 minutes of prices based on past prices.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the price time series data")
    args = parser.parse_args()

    while True:
        # Load data from the specified CSV file
        data = load_data_from_csv(args.csv_file)

        # Create a time series dataset for predicting the next 10 minutes (10 steps)
        window_size = 10
        prediction_steps = 10
        X, y = create_time_series_dataset(data, window_size, prediction_steps)

        # Normalize the data
        scaler = StandardScaler()

        # Fit and transform the scaler on the data
        X = scaler.fit_transform(X)

        # Reshape input data to be 3D (batch_size, time_steps, input_dim)
        input_shape = (window_size, 1)  # One feature: past price_usd values
        X = X.reshape((-1, *input_shape))

        # Create and train the RNN model
        model = create_rnn_model(input_shape, prediction_steps)
        model.fit(X, y, epochs=1000, batch_size=32)

        # Use the trained model to make predictions for the next 10 minutes
        last_window = X[-1]  # Take the last window from the data
        predicted_prices = predict_next_steps(model, last_window, prediction_steps)

        print("Predicted prices for the next 10 minutes:")
        for i, price in enumerate(predicted_prices):
            print(f"Minute {i+1}: {price:.2f}")

        # Save the predictions to a CSV file
        with open("./predictions.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for price in predicted_prices:
                print(price)
                writer.writerow([price])
            file.close()

        time.sleep(60)