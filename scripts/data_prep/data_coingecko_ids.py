import pandas as pd
import requests

def get_coingecko_id(coin_ticker):
    url = f"https://www.coingecko.com/en/search_v2?query={coin_ticker}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        coins = data.get('coins', [])
        if coins:
            return coins[0].get('id', 'UNKNOWN')
        else:
            return 'UNKNOWN'
    except requests.RequestException:
        return 'UNKNOWN'

def map_coinbase_to_coingecko(csv_path):
    df = pd.read_csv(csv_path)
    filtered_df = df[df['Exchange'] == 1]

    coinbase_to_coingecko = {}
    for idx, coin_ticker in enumerate(filtered_df['Asset']):
        if idx < 145:
            continue
        coingecko_id = get_coingecko_id(coin_ticker)
        print("{}/{}: {}".format(idx, len(filtered_df), coingecko_id))
        coinbase_to_coingecko[coin_ticker] = coingecko_id

    return coinbase_to_coingecko

# Example usage
csv_path = './data/coinbase_coins.csv'
result = map_coinbase_to_coingecko(csv_path)
print(result)
