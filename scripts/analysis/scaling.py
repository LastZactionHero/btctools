import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_rows(arr):
    return scale_rows_delta(arr)


def scale_rows_delta(arr):
    scaled_data = []
    for row in arr:
        b = np.delete(row, 0)
        a = np.delete(row, len(row)-1)
        scaled_row = (b - a) / a
        scaled_data.append(scaled_row)
    return np.array(scaled_data)


def scale_rows_standard(arr):
    scaler = StandardScaler()
    scaled_data = []
    for row in arr:
        scaled_row = scaler.fit_transform(row.reshape(-1, 1)).flatten()
        scaled_data.append(scaled_row)
    return np.array(scaled_data)


def scale_rows_min_max(arr):
    scaled_data = []
    for row in arr:
        min_maxed_row = row / row.max()
        scaled_data.append(min_maxed_row)
    return np.array(scaled_data)
