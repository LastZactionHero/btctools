import numpy as np
import pandas as pd
import os
from sympy import N

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class BuyerPredictionModel:
    MODEL_FILENAME = "./models/lstm_series_240m.h5"
    SEQUENCE_LOOKBEHIND_MINUTES = 240
    PREDICTION_LOOKAHEAD_MINUTES = 30

    def __init__(self, timesource, cache=None):
        self.model = load_model(self.MODEL_FILENAME)
        self.cache = cache
        self.timesource = timesource

    def predict(self, data, latest=False):
        data_no_timestamps = data.copy().drop("timestamp", axis=1)

        if self.cache:
            model_name = os.path.basename(self.MODEL_FILENAME).split(".")[0]
            cached_predictions = self.cache.load_predictions_from_cache(model_name, self.timesource.now())
            if cached_predictions is not None:
                return cached_predictions

        timestamp = self.timesource.now()

        row_idx = len(data) - 1
        if latest is not True:
            row_idx = self.find_row_idx(data, timestamp)
        predictions = self.build_predictions(data_no_timestamps, row_idx)

        if self.cache:
            model_name = os.path.basename(self.MODEL_FILENAME).split(".")[0]
            self.cache.save_to_cache(model_name, timestamp, predictions)
        return predictions

    def find_row_idx(self, data, timestamp):
        return data['timestamp'].searchsorted(timestamp) - 1
    
    def build_predictions(self, data_no_timestamps, row_idx):
        predict_sequences = data_no_timestamps.loc[list(range(row_idx - self.SEQUENCE_LOOKBEHIND_MINUTES, row_idx, 1))]

        coins = []
        scalers = []
        predict_sequences_scaled = []

        for i, coin in enumerate(predict_sequences):
            scaler = MinMaxScaler()
            ps = scaler.fit_transform(predict_sequences[coin].values.reshape(1,-1).T).flatten()
            
            coins.append(coin)
            scalers.append(scaler)
            predict_sequences_scaled.append(ps)

        predictions_scaled = self.model.predict(np.array(predict_sequences_scaled))

        predictions = pd.DataFrame(columns=['Coin', 'Latest', 'Max', 'Max Delta', 'Min', 'Min Delta'])
        for i, ps in enumerate(predictions_scaled):
            latest_price = data_no_timestamps.iloc[row_idx,i]

            unscaled = scalers[i].inverse_transform(ps.reshape(1, -1).T).flatten()
            p_max = unscaled.max()
            p_min = unscaled.min()

            d_max = round((p_max - latest_price) / latest_price * 100.0, 2)
            d_min = round((p_min - latest_price) / latest_price * 100.0, 2)
            d_mean = round((np.mean(unscaled) - latest_price) / latest_price * 100, 2)

            row = pd.DataFrame({
                'Coin': [coins[i]],
                'Latest': latest_price,
                'Max': [p_max],
                'Max Delta': [d_max],
                'Min': [p_min],
                'Min Delta': [d_min],
                'Mean Delta': d_mean
            })
            predictions = pd.concat([predictions, row])

        predictions = predictions.sort_values(by="Max Delta", ascending=False)
        return predictions