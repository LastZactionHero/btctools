import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class BuyerPredictionModel:
    MODEL_FILENAME = "./models/lstm_series_240m.h5"
    SEQUENCE_LOOKBEHIND_MINUTES = 240
    PREDICTION_LOOKAHEAD_MINUTES = 30

    def __init__(self, data, cache=None):
        self.data = data
        self.data_no_timestamps = self.data.copy().drop("timestamp", axis=1)
        self.model = load_model(self.MODEL_FILENAME)
        self.cache = cache

    def predict(self, timestamp):
        if self.cache:
            model_name = os.path.basename(self.MODEL_FILENAME).split(".")[0]
            cached_predictions = self.cache.load_predictions_from_cache(model_name, timestamp)
            if cached_predictions is not None:
                return cached_predictions

        row_idx = self.find_row_idx(timestamp)
        predictions = self.build_predictions(row_idx)

        if self.cache:
            model_name = os.path.basename(self.MODEL_FILENAME).split(".")[0]
            self.cache.save_to_cache(model_name, timestamp, predictions)
        return predictions

    def find_row_idx(self, timestamp):
        return self.data['timestamp'].searchsorted(timestamp) - 1
    
    def build_predictions(self, row_idx):
        predict_sequences = self.data_no_timestamps.loc[list(range(row_idx - self.SEQUENCE_LOOKBEHIND_MINUTES, row_idx, 1))]

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
            latest_price = self.data_no_timestamps.iloc[row_idx,i]

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