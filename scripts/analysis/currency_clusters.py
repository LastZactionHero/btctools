import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


# Load the correlation matrix
data = pd.read_csv("./data/correlation_matrix.csv", index_col=0)
correlation_matrix = data.values

# Perform agglomerative clustering
clustering = AgglomerativeClustering(n_clusters=100, affinity='correlation', linkage='average')
clustering.fit(correlation_matrix)

# Get the cluster labels
labels = clustering.labels_

# Print the cluster assignments
# print("Cluster assignments:")
# print("Bitcoin:", labels[0])
# print("Ethereum:", labels[1])
# print("Tether:", labels[2])
# print("Ripple:", labels[3])

# Plot the dendrogram
# plt.figure()
# plt.title("Dendrogram")
# plt.dendrogram(clustering.children_, labels=range(len(labels)))
# plt.show()

print(labels)

import numpy as np

# Create a dictionary to store the clusters
clusters = {}

# Iterate over the labels and add each coin to its corresponding cluster
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(data.columns[i])

# Print the clusters
for cluster_id, coins in clusters.items():
    print(f"{cluster_id}: {', '.join(coins)}")