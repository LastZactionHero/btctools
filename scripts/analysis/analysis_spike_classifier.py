import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from scaling import scale_rows

output_file = sys.argv[1]

# Load and preprocess data
def load_data(filename, label):
    df = pd.read_csv(filename, header=0)
    df['label'] = label
    return df

# Load data
rise_series_fake = load_data('data/rise_series_fake.csv', 0)
rise_series_real = load_data('data/rise_series_real.csv', 1)

# Combine and shuffle data
data = pd.concat([rise_series_fake, rise_series_real]).sample(frac=1).reset_index(drop=True)

# Split features and labels
X = data.drop('label', axis=1).values
y = data['label'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data (important for neural network models)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
X_train = scale_rows(X_train)
X_test = scale_rows(X_test)

# Reshape data for LSTM layer
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Create model
model = Sequential([
    # LSTM(50, input_shape=(X_train.shape[1], 1)),
    # Dense(1, activation='sigmoid')
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

model.save(output_file)