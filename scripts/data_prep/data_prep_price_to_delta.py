import numpy as np
import pandas as pd
import sys

filename = sys.argv[1]
output_filename = sys.argv[2]
pd_data = pd.read_csv(filename)
data = pd_data.to_numpy()

percentage_increase = (data[1:, 1:] - data[:-1, 1:]) / data[:-1, 1:] * 100

timestamps = data[1:, 0]
timestamps = timestamps.reshape(-1, 1)
combined_data = np.hstack((timestamps, percentage_increase))

# print(combined_data)
final_data = pd.DataFrame(combined_data, columns=['timestamp'] + pd_data.columns[1:].tolist())
final_data.iloc[:, 1:] = final_data.iloc[:, 1:].round(2)
final_data.to_csv(output_filename, index=False)
