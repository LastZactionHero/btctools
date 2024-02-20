import requests
import pandas as pd

class CryptoExchangeRatesFetcher:
    def __init__(self, url, filename, logger):
        self.url = url
        self.filename = filename
        self.logger = logger

    def fetch(self, cached=False):
        if cached:
            self.logger.info(f"Using cached data from {self.filename}")
            
        else:
            try:
                # Attempt to download the Crypto Exchange Rates CSV
                self.logger.info(f"Attempting to download Crypto Exchange Rates from {self.url}")
                response = requests.get(self.url)
                response.raise_for_status()  # Raises a HTTPError for unsuccessful status codes.

                # Save the CSV content to specified filename
                with open(self.filename, 'wb') as file:
                    file.write(response.content)
                
                self.logger.info(f"Successfully downloaded and saved the Crypto Exchange Rates CSV to {self.filename}.")

            except requests.exceptions.HTTPError as errh:
                self.logger.error(f"Http Error: {errh}")
            except requests.exceptions.ConnectionError as errc:
                self.logger.error(f"Error Connecting: {errc}")
            except requests.exceptions.Timeout as errt:
                self.logger.error(f"Timeout Error: {errt}")
            except requests.exceptions.RequestException as err:
                self.logger.error(f"Oops: Something Else: {err}")

        try:
            # Attempt to read the CSV file to a pandas DataFrame
            self.logger.info(f"Loading Crypto Exchange Rates data from {self.filename}")
            data = pd.read_csv(self.filename)
            self.logger.info(f"Data loaded successfully from {self.filename}")
            return data
        except pd.errors.EmptyDataError:
            self.logger.error(f"No data found in {self.filename}. Please ensure the file is not empty.")
        except FileNotFoundError:
            self.logger.error(f"File {self.filename} not found. Please ensure the path is correct and the file exists.")
        except Exception as e:
            self.logger.error(f"An error occurred while loading data from {self.filename}: {e}")
            raise
