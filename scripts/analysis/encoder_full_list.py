import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, MultiHeadAttention, LayerNormalization, Dropout, Reshape
# from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
import datetime
import pytz


def convert_unix_to_mountain_time(unix_timestamp):
    # Create a timezone object for Mountain Time
    mountain_time = pytz.timezone('America/Denver')

    # Convert the Unix timestamp to a datetime object
    dt_utc = datetime.datetime.utcfromtimestamp(unix_timestamp)

    # Convert the UTC datetime to Mountain Time
    dt_mountain = dt_utc.astimezone(mountain_time)

    return dt_mountain

data = pd.read_csv(sys.argv[1], index_col=0)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Function to create 30-minute windows
def create_windows(data, window_size=30):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

# Create the 30-minute windows
# For the example, I'm using a window size of 3 due to limited data
window_size = 30  # Change this to 30 for actual 30-minute windows
windows = create_windows(data_scaled, window_size)

# The target can be the next window's data or some other representation
# For simplicity, I'm using the same window as target
X = windows
y = windows  # In a real case, define a proper target

# Define the input shape
input_shape = (window_size, data.shape[1])
embedding_size = 20  # Size of the embedding vector

# Input layer
inputs = Input(shape=input_shape)

# Encoder layers
x = MultiHeadAttention(num_heads=2, key_dim=embedding_size)(inputs, inputs)
x = LayerNormalization(epsilon=1e-6)(x)
x = Dropout(0.1)(x)
x = Flatten()(x)
encoder_output = Dense(embedding_size, activation='relu')(x)

# Create the encoder model
encoder_model = Model(inputs=inputs, outputs=encoder_output)

# Additional layers for the full model (decoder part)
x = Dense(30 * 253, activation='linear')(encoder_output)  # Adjust the number of units
outputs = Reshape((30, 253))(x)  # Reshape to [30, 253]

# Create the full model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the full model
history = model.fit(X, y, epochs=10, batch_size=32)

# Save only the encoder model
encoder_model.save('./models/encoder_only.h5')

encoder_model.predict(np.expand_dims(windows[0], axis=0))


embeddings = encoder_model.predict(windows)

from sklearn.cluster import KMeans


# Number of clusters
n_clusters = 10

# Train KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=43).fit(embeddings)

cluster_assignments = kmeans.labels_


# For a new embedding vector
# new_embedding = encoder_model.predict(new_window)  # Replace 'new_window' with actual data
# new_cluster_assignment = kmeans.predict(new_embedding)


# kmeans.predict(encoder_model.predict(np.expand_dims(windows[0], axis=0)))
for cluster_id in range(10):
    cluster_windows = windows[cluster_assignments == cluster_id]

    inverse_transformed_windows = []
    for window in cluster_windows:
        inverse_transformed_window = scaler.inverse_transform(window)
        inverse_transformed_windows.append(inverse_transformed_window)
    inverse_transformed_windows = np.array(inverse_transformed_windows)

    average_cluster = np.mean(inverse_transformed_windows, axis=1)
    row_sums = np.sum(average_cluster, axis=0).round(4)
    print("Cluster: {}".format(cluster_id))
    print(row_sums)

# import pdb; pdb.set_trace()




# Example usage
# unix_timestamp = 1701815272  # Replace with your timestamp
# dt_mountain = convert_unix_to_mountain_time(unix_timestamp)
# print(dt_mountain)
