import csv
import time

# Initial balances
bitcoin_balance = 1.0  # Initial Bitcoin balance
usd_balance = 0  # Initial USD balance

# Minimum threshold for buy/sell alerts (default to $10)
min_threshold = 10.0

# Function to read the latest price from bitcoin_exchange_rate.csv
def read_latest_price():
    with open('./bitcoin_exchange_rate.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamp = int(row['timestamp'])
            price_usd = float(row['price_usd'])
    return timestamp, price_usd

# Function to read the predicted price from predictions.csv
def read_predicted_price():
    with open('./predictions.csv', 'r') as file:
        lines = file.readlines()
        if lines:
            predicted_price = float(lines[-1])
        else:
            predicted_price = None
    return predicted_price

# Function to check and execute buy/sell recommendations
def check_and_execute_trade(current_price_btc_usd, predicted_price_btc_usd):
    global bitcoin_balance
    global usd_balance

    if predicted_price_btc_usd is not None:
        # Check for sell recommendation
        if bitcoin_balance > 0:
            bitcoin_holdings = bitcoin_balance * current_price_btc_usd
            predicted_bitcoin_holdings = bitcoin_balance * predicted_price_btc_usd
            predicted_delta = predicted_bitcoin_holdings - bitcoin_holdings
            print("10 minute prediction: {} {} {}".format(bitcoin_holdings, predicted_bitcoin_holdings, predicted_delta))

            if predicted_delta < -1 * min_threshold:
                print("Recommendation: Sell!")

        # Check for buy recommendation
        if usd_balance > 0:
            bitcoin_holdings = usd_balance / current_price_btc_usd
            predicted_bitcoin_holdings = bitcoin_holdings * predicted_price_btc_usd
            predicted_delta = predicted_bitcoin_holdings - usd_balance

            print("10 minute prediction: {} {} {}".format(usd_balance, predicted_bitcoin_holdings, predicted_delta))

            if predicted_delta > min_threshold:
                print("Recommendation: Buy!")

# Main loop to continuously monitor and make recommendations
if __name__ == "__main__":
    while True:
        timestamp, current_price_usd = read_latest_price()
        current_price_btc_usd = current_price_usd  # Assuming 1 Bitcoin = current_price_usd USD

        predicted_price_btc_usd = read_predicted_price()

        if predicted_price_btc_usd is not None:
            print(f"Timestamp: {timestamp}")
            print(f"Current BTC/USD Price: {current_price_btc_usd:.2f}")
            print(f"Predicted BTC/USD Price (10 minutes later): {predicted_price_btc_usd:.2f}")

            check_and_execute_trade(current_price_btc_usd, predicted_price_btc_usd)
        
        time.sleep(60)
