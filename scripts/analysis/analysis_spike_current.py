import sys
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

PREFETCH_MINUTES = 60

def scale_rows(arr):
    scaler = StandardScaler()
    scaled_data = []

    for row in arr:
        scaled_row = scaler.fit_transform(row.reshape(-1, 1)).flatten()
        scaled_data.append(scaled_row)

    return np.array(scaled_data)


loaded_model = load_model(sys.argv[1])
data = pd.read_csv(sys.argv[2])

prefetch_series = data[len(data) - PREFETCH_MINUTES : len(data)].T.to_numpy()
prefetch_series_scaled = scale_rows(prefetch_series)

predictions = loaded_model.predict(prefetch_series_scaled)

for idx, prediction in enumerate(predictions):
    coin = data.columns[idx]
    if prediction[0] > 0.9:
        print("\n** Maybe!: {} {}".format(prediction[0], coin))