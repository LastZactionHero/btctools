import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.cluster import KMeans
from scaling import scale_rows

# Load the data
df = pd.read_csv("./data/crypto_exchange_rates.csv")

# Load the encoder model
encoder = load_model("./data/encoder.h5")

# Define the window size
window_size = 30

# Create a list to store the embeddings
embeddings = []

# Iterate over the DataFrame in windows of 30 minutes
# for i in range(0, len(df) - window_size + 1, window_size):
#     # Extract a window of data
#     window = df.iloc[i:i + window_size]

#     # Scale the rows of the window

#     window = window.drop("timestamp", axis=1)
#     for columm in window:
#         window_scaled = 10 * scale_rows([window[columm]])
#         embedding = encoder.predict(window_scaled)
#         embeddings.append(embedding)

# Creating Windows
windows = []
for coin in df.columns:
    if coin == "timestamp":
        next

    coin_data = df[coin].values
    for i in range(len(coin_data) - window_size + 1):
        window = coin_data[i: i + window_size]
        window = scale_rows([window])[0] * 10
        windows.append(window)

import pdb; pdb.set_trace()
# # Concatenate all of the embeddings into a single array
# embeddings_array = np.concatenate(embeddings)

# # Run K-means clustering on the embeddings
# kmeans = KMeans(n_clusters=10)
# kmeans.fit(embeddings_array)

# # Save the K-means model
# kmeans.save("./data/encoder_kmeans.h5")