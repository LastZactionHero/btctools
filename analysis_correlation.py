import pandas as pd

# Load your data
# Replace 'file_path' with the path to your CSV file
file_path = './data/crypto_exchange_delta.csv'
data = pd.read_csv(file_path)

# Set the timestamp as the index, if it's not already
data.set_index('timestamp', inplace=True)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print(correlation_matrix)

correlation_matrix.to_csv("./data/correlation_matrix.csv")