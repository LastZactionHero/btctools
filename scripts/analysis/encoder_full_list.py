from sklearn.cluster import KMeans
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
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

data = pd.read_csv(sys.argv[1], index_col=0)
timestamps = pd.read_csv(sys.argv[1])['timestamp']
trending_together = ["1inch",
                     "aave",
                     "alchemy-pay",
                     "cardano",
                     "adventure-gold",
                     "alchemix",
                     "algorand",
                     "my-neighbor-alice",
                     "ankr",
                     "aragon",
                     "apecoin",
                     "api3",
                     "aptos",
                     "arbitrum",
                     "arpa",
                     "astar",
                     "automata",
                     "cosmos",
                     "auction",
                     "audius",
                     "avalanche-2",
                     "axie-infinity",
                     "badger-dao",
                     "balancer",
                     "band-protocol",
                     "basic-attention-token",
                     "bitcoin-cash",
                     "biconomy",
                     "bitcoin",
                     "blur",
                     "bancor",
                     "barnbridge",
                     "bitcoin.1",
                     "coin98",
                     "celer-network",
                     "chiliz",
                     "clover-finance",
                     "internet-computer",
                     "coti",
                     "crypto-com-chain",
                     "curve-dao-token",
                     "cartesi",
                     "convex-finance",
                     "dash",
                     "dash.1",
                     "dogecoin",
                     "polkadot",
                     "enjincoin",
                     "ethereum-name-service",
                     "eos",
                     "ethereum-classic",
                     "ethereum",
                     "harvest-finance",
                     "fetch-ai",
                     "filecoin",
                     "stafi",
                     "flow",
                     "frax-share",
                     "gala",
                     "gala.1",
                     "aavegotchi",
                     "moonbeam",
                     "stepn",
                     "gnosis",
                     "gods-unchained",
                     "the-graph",
                     "gitcoin",
                     "hedera-hashgraph",
                     "hashflow",
                     "internet-computer.1",
                     "aurora-dao",
                     "illuvium",
                     "immutable-x",
                     "injective-protocol",
                     "iotex",
                     "jasmycoin",
                     "kava",
                     "kyber-network-crystal",
                     "kusama",
                     "lido-dao",
                     "chainlink",
                     "litecoin",
                     "league-of-kingdoms",
                     "loom-network-new",
                     "livepeer",
                     "liquity",
                     "loopring",
                     "liquid-staked-ethereum",
                     "litecoin.1",
                     "magic",
                     "decentraland",
                     "mask-network",
                     "matic-network",
                     "measurable-data-token",
                     "metis-token",
                     "maker",
                     "msol",
                     "near",
                     "nkn",
                     "numeraire",
                     "ocean-protocol",
                     "origin-protocol",
                     "omisego",
                     "ooki",
                     "optimism",
                     "orchid-protocol",
                     "perpetual-protocol",
                     "playdapp",
                     "matic-network.1",
                     "near.1",
                     "vulcan-forged",
                     "quant-network",
                     "quickswap",
                     "render-token",
                     "request-network",
                     "iexec-rlc",
                     "render-token.1",
                     "oasis-network",
                     "rocket-pool",
                     "the-sandbox",
                     "shiba-inu",
                     "skale",
                     "havven",
                     "solana",
                     "space-id",
                     "spell-token",
                     "stargate-finance",
                     "storj",
                     "blockstack",
                     "sui",
                     "superfarm",
                     "sushi",
                     "havven.1",
                     "big-time",
                     "tellor",
                     "uma",
                     "unifi-protocol-dao",
                     "uniswap",
                     "ethos",
                     "voxies",
                     "wrapped-bitcoin",
                     "stellar",
                     "ripple",
                     "tezos",
                     "yearn-finance",
                     "yfii-finance",
                     "zcash",
                     "zencash",
                     "0x"]
data = data[trending_together]

# Normalize the data
scaler = MinMaxScaler(feature_range=(-10, 10))
data_scaled = scaler.fit_transform(data)

# Function to create 30-minute windows


def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)


# Create the 30-minute windows
# For the example, I'm using a window size of 3 due to limited data
window_size = 120  # Change this to 30 for actual 30-minute windows
windows = create_windows(data_scaled, window_size)

# The target can be the next window's data or some other representation
# For simplicity, I'm using the same window as target
X = windows
y = windows  # In a real case, define a proper target

# Define the input shape
# input_shape = (window_size, data.shape[1])
# embedding_size = 20  # Size of the embedding vector

# # Input layer
# inputs = Input(shape=input_shape)

# # Encoder layers
# x = MultiHeadAttention(num_heads=2, key_dim=embedding_size)(inputs, inputs)
# x = LayerNormalization(epsilon=1e-6)(x)
# x = Dropout(0.1)(x)
# x = Flatten()(x)
# encoder_output = Dense(embedding_size, activation='relu')(x)

# # Create the encoder model
# encoder_model = Model(inputs=inputs, outputs=encoder_output)

# # Additional layers for the full model (decoder part)
# # Adjust the number of units
# x = Dense(window_size * data.shape[1], activation='linear')(encoder_output)
# outputs = Reshape((window_size, data.shape[1]))(x)  # Reshape to [30, 253]

# # Create the full model
# model = Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer='adam', loss='mse')

# history = model.fit(X, y, epochs=10, batch_size=32)

# # Save only the encoder model
# encoder_model.save('./models/encoder_only.h5')

# embeddings = encoder_model.predict(windows)

# Define the input shape
input_shape = (window_size, data.shape[1])
embedding_size = 20  # Size of the embedding vector

# Input layer
inputs = Input(shape=input_shape)

# Replacing the encoder layers with Dense layers
x = Flatten()(inputs)  # Flatten the input
x = Dense(embedding_size, activation='relu')(x)
x = Dropout(0.1)(x)
encoder_output = Dense(embedding_size, activation='relu')(x)

# Create the encoder model
encoder_model = Model(inputs=inputs, outputs=encoder_output)

# Additional layers for the full model (decoder part)
# Adjust the number of units
x = Dense(window_size * data.shape[1], activation='linear')(encoder_output)
outputs = Reshape((window_size, data.shape[1]))(x)  # Reshape to [30, 253]

# Create the full model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=10, batch_size=32)

# Save only the encoder model
encoder_model.save('./models/encoder_only.h5')

embeddings = encoder_model.predict(windows)


# Number of clusters
n_clusters = 10

# Train KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=43).fit(embeddings)

cluster_assignments = kmeans.labels_

clusters = {}

for cluster_id in range(n_clusters):
    cluster_windows = windows[cluster_assignments == cluster_id]
    if(len(cluster_windows) == 0):
        continue

    inverse_transformed_windows = []
    for window in cluster_windows:
        inverse_transformed_window = scaler.inverse_transform(window)
        inverse_transformed_windows.append(inverse_transformed_window)
    inverse_transformed_windows = np.array(inverse_transformed_windows)

    average_cluster = np.mean(inverse_transformed_windows, axis=1)
    row_sums = np.sum(average_cluster, axis=0).round(4)

    clusters[cluster_id] = {
        'average': np.mean(row_sums).round(4),
        'stdev': np.std(row_sums).round(4),
        'row_sums': row_sums
    }
    

from datetime import datetime
def convert_unix_to_readable(timestamp):
    # Convert Unix timestamp to datetime object
    date_time = datetime.fromtimestamp(timestamp)
    
    # Format the datetime object to a string (e.g., "YYYY-MM-DD HH:MM:SS")
    readable_string = date_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return readable_string


for idx, cluster_assignment in enumerate(cluster_assignments):
    timestamp = timestamps.iloc[min(idx, len(timestamps) - 1)]
    if(datetime.fromtimestamp(timestamp).minute % window_size != 0):
        continue

    readable_timestamp = convert_unix_to_readable(timestamp)

    cluster = clusters[cluster_assignment]
    print("{}:\t{}\t{}\t{}".format(readable_timestamp, cluster_assignment, cluster['average'], cluster['stdev']))

lstm_scaler = MinMaxScaler(feature_range=(0, 1))
cluster_scaled = lstm_scaler.fit_transform(cluster_assignments.reshape(-1, 1))

sequence_length = 24
generator = TimeseriesGenerator(cluster_scaled, cluster_scaled, length=sequence_length, batch_size=1)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(generator, epochs=5, verbose=1)

last_sequence = cluster_scaled[-sequence_length:] 

for i in range(60):
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    predicted = model.predict(last_sequence, verbose=0)

    predicted_inverted = lstm_scaler.inverse_transform(predicted)
    cluster_assignment = round(predicted_inverted[0][0])

    cluster = clusters[cluster_assignment]
    print("{}:\t{}\t{}\t{}".format(i, cluster_assignment, cluster['average'], cluster['stdev']))

    last_sequence = np.append(last_sequence, predicted)[1:]

    

# import pdb; pdb.set_trace()
# # predicted = lstm_scaler.inverse_transform(predicted)

# # next_cluster = round(predicted[0][0])
# # l