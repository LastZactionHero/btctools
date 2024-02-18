import requests
import pandas as pd

class CryptoExchangeRatesFetcher:
    def __init__(self, URL, FILENAME):
        self.url = URL
        self.filename = FILENAME

    def fetch(self, cached=False):
        try:
            # Download the Crypto Exchange Rates CSV
            if cached is False:
                response = requests.get(self.url)
                response.raise_for_status()  # Raises a HTTPError if the HTTP request returned an unsuccessful status code.

                # Save the CSV content to FILENAME_CRYPTO_EXCHANGE_RATES
                with open(self.filename, 'wb') as file:
                    file.write(response.content)
            print("Successfully downloaded and saved the Crypto Exchange Rates CSV.")
        except requests.exceptions.HTTPError as errh:
            print(f"Http Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Oops: Something Else: {err}")

        return pd.read_csv(self.filename)