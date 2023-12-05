import requests
import csv
import time
import os

def get_crypto_to_usd_exchange_rates(ids):
    # CoinGecko API endpoint for multiple cryptocurrencies to USD exchange rates
    endpoint = "https://api.coingecko.com/api/v3/simple/price"
    
    # Parameters for the API request
    params = {
        "ids": ",".join(ids),  # Join the cryptocurrency IDs with a comma
        "vs_currencies": "usd"
    }
    
    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_csv(timestamp, exchange_rates, filename, write_headers=False):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # if not file_exists or write_headers:
        #     headers = ['timestamp'] + ids
        #     writer.writerow(headers)
        
        row = [timestamp] + [exchange_rates[crypto_id]['usd'] for crypto_id in ids]
        writer.writerow(row)

if __name__ == "__main__":
    filename = "crypto_exchange_rates.csv"
    ids = ["bitcoin", "ethereum", "dogecoin", "litecoin", "ripple", "cardano", "polkadot", "stellar", "chainlink", "shiba-inu", "solana"]
    while True:
        timestamp = int(time.time())
        exchange_rates = get_crypto_to_usd_exchange_rates(ids)
        
        if exchange_rates is not None:
            print(f"Timestamp: {timestamp}, Exchange rates: {exchange_rates}")
            save_to_csv(timestamp, exchange_rates, filename, write_headers=True)
        else:
            print("Failed to fetch the exchange rates.")
        
        # Wait for one minute before the next iteration
        time.sleep(60)