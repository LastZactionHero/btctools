import requests
import csv
import time

def get_bitcoin_to_usd_exchange_rate():
    # CoinGecko API endpoint for Bitcoin to USD exchange rate
    endpoint = "https://api.coingecko.com/api/v3/simple/price"
    
    # Parameters for the API request
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd"
    }
    
    try:
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        # Extract the Bitcoin to USD exchange rate from the response
        bitcoin_to_usd_rate = data["bitcoin"]["usd"]
        
        return bitcoin_to_usd_rate
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_csv(timestamp, price, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, price])

if __name__ == "__main__":
    filename = "bitcoin_exchange_rate.csv"

    while True:
        timestamp = int(time.time())
        exchange_rate = get_bitcoin_to_usd_exchange_rate()
        
        if exchange_rate is not None:
            print(f"Timestamp: {timestamp}, Bitcoin to USD exchange rate: ${exchange_rate}")
            save_to_csv(timestamp, exchange_rate, filename)
        else:
            print("Failed to fetch the exchange rate.")
        
        # Wait for one minute before the next iteration
        time.sleep(60)
