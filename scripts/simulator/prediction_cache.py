import os
import pandas as pd

class PredictionCache():
    @staticmethod
    def load_predictions_from_cache(model_name, timestamp):
        cache_file_path = f"./cache/{model_name}_{timestamp}.csv"

        if os.path.exists(cache_file_path):
            predictions = pd.read_csv(cache_file_path)
            return predictions
        return None

    @staticmethod
    def save_to_cache(model_name, timestamp, predictions):
        cache_file_path = f"./cache/{model_name}_{timestamp}.csv"

        predictions.to_csv(cache_file_path, index=False)
