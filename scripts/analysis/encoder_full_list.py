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


def convert_unix_to_mountain_time(unix_timestamp):
    # Create a timezone object for Mountain Time
    mountain_time = pytz.timezone('UTC')

    # Convert the Unix timestamp to a datetime object
    dt_utc = datetime.datetime.utcfromtimestamp(unix_timestamp)

    # Convert the UTC datetime to Mountain Time
    dt_mountain = dt_utc.astimezone(mountain_time)

    return dt_mountain


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
# Adjust the number of units
x = Dense(30 * data.shape[1], activation='linear')(encoder_output)
outputs = Reshape((30, data.shape[1]))(x)  # Reshape to [30, 253]

# Create the full model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the full model
history = model.fit(X, y, epochs=10, batch_size=32)

# Save only the encoder model
encoder_model.save('./models/encoder_only.h5')

encoder_model.predict(np.expand_dims(windows[0], axis=0))


embeddings = encoder_model.predict(windows)


# Number of clusters
n_clusters = 6

# Train KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=43).fit(embeddings)

cluster_assignments = kmeans.labels_


# For a new embedding vector
# new_embedding = encoder_model.predict(new_window)  # Replace 'new_window' with actual data
# new_cluster_assignment = kmeans.predict(new_embedding)


# kmeans.predict(encoder_model.predict(np.expand_dims(windows[0], axis=0)))
for cluster_id in range(n_clusters):
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
np.set_printoptions(threshold=np.inf)
print(cluster_assignments)

# for idx, cluster in enumerate(cluster_assignments):
#     t = convert_unix_to_mountain_time(timestamps.iloc[idx])
#     print("{}: {}".format(t, cluster))