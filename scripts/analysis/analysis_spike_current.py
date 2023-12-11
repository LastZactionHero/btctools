import sys
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from scaling import scale_rows

PREFETCH_MINUTES = 30

loaded_model = load_model(sys.argv[1])
data = pd.read_csv(sys.argv[2])

prefetch_series = data[len(data) - PREFETCH_MINUTES : len(data)].T.to_numpy()
prefetch_series_scaled = scale_rows(prefetch_series)

predictions = loaded_model.predict(prefetch_series_scaled)

max_prediction = max(predictions)


for idx, prediction in enumerate(predictions):
    coin = data.columns[idx]
    if prediction[0] > 0.7:
        print("** Maybe!: {} {}".format(prediction[0], coin))
    elif prediction[0] == max_prediction:
        print("Max: {} {}".format(prediction[0], coin))
    # else:
    #     print("** Nope!: {} {}".format(prediction[0], coin))