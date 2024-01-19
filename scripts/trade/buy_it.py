import datetime
import numpy as np
import pandas as pd
import re
import requests
import time
from coinbaseadvanced.client import CoinbaseAdvancedTradeAPIClient, Side
from prettytable import PrettyTable

PRIOR_MINUTES_TO_REVIEW = 14 * 24 * 60 # 14 days

api_key_name = os.getenv("COINBASE_API_KEY_NAME")
private_key = os.getenv("COINBASE_PRIVATE_KEY")
coinbaseClient = CoinbaseAdvancedTradeAPIClient.from_cloud_api_keys(api_key_name, private_key)

data = pd.read_csv("/content/crypto_exchange_rates.csv")
predictions = pd.read_csv("/content/lstm_predictions.csv")

positive_predictions = predictions[predictions['delta'] > 0.05]

def hit_counts_in_range(coin, data, prior_minutes):
  prior_values = data[coin][len(data) - prior_minutes : len(data) - 1].values

  hit_count = 0
  skip_until = 0

  for idx, value in enumerate(prior_values):
    if(idx < skip_until):
      continue

    next_values = prior_values[idx:]
    max_subsequence_gain = (next_values.max() - value) / value

    if (max_subsequence_gain < 0.04):
      continue

    for next_idx, next_value in enumerate(next_values):
      gain = (next_value - value) / value
      if gain > 0.04:
        skip_until = idx + next_idx
        hit_count += 1
        break
  return hit_count

def percent_above_hit_percent(coin, data, prior_minutes):
  prior_values = data[coin][len(data) - prior_minutes : len(data) - 1].values
  latest_price = prior_values[-1]
  minutes_above_hit_price = len(prior_values[prior_values > latest_price * 1.04])

  return minutes_above_hit_price / prior_minutes

  columns = ['Symbol', 'Coin', 'Latest Price', 'Predicted Delta', 'Hit Count', 'Min Above']
df = pd.DataFrame(columns=columns)

for index, prediction in positive_predictions.iterrows():
  coin = prediction[0]
  symbol = gecko_coinbase_currency_map[coin]

  if symbol == 'UNSUPPORTED':
    continue

  hit_count = hit_counts_in_range(coin, data, PRIOR_MINUTES_TO_REVIEW)
  percent_above = percent_above_hit_percent(coin, data, PRIOR_MINUTES_TO_REVIEW)

  row = [gecko_coinbase_currency_map[coin], coin, data[coin].values[-1], prediction['delta'], hit_count, percent_above]
  df = df.append(pd.Series(row, index=columns), ignore_index=True)

columns = ['Symbol', 'Coin', 'Latest Price', 'Predicted Delta', 'Hit Count', 'Min Above']
df = pd.DataFrame(columns=columns)

for index, prediction in positive_predictions.iterrows():
  coin = prediction[0]
  symbol = gecko_coinbase_currency_map[coin]

  if symbol == 'UNSUPPORTED':
    continue

  hit_count = hit_counts_in_range(coin, data, PRIOR_MINUTES_TO_REVIEW)
  percent_above = percent_above_hit_percent(coin, data, PRIOR_MINUTES_TO_REVIEW)

  row = [gecko_coinbase_currency_map[coin], coin, data[coin].values[-1], prediction['delta'], hit_count, percent_above]
  df = df.append(pd.Series(row, index=columns), ignore_index=True)

df_filtered = df[(df['Min Above'] >= 0.35) & (df['Min Above'] <= 0.5)]
df_filtered = df_filtered[(df_filtered['Predicted Delta'] >= 0.09)]
df_filtered = df_filtered[(df_filtered['Hit Count'] >= 5)]
df_filtered = df_filtered.sort_values(by='Min Above')
print(df_filtered)

accounts = coinbaseClient.list_accounts(limit=250)

def fetch_portfolio(accounts):
  MIN_BALANCE_USD = 3.0 # Only show coins with more than this amount

  portfolio = []
  balance_usdc = 0

  for account in accounts:
    balance_currency = float(account.available_balance.value)
    if balance_currency > 0:
      currency = account.currency

      balance_usd = 0.0

      if currency != 'USDC':
        product = f"{currency}-USDC"
        best_bid_ask = coinbaseClient.get_best_bid_ask([product])
        exchange_rate_usd = float(best_bid_ask.pricebooks[0].bids[0].price)
        balance_usd = round(balance_currency * exchange_rate_usd, 2)
      else:
        balance_usdc = round(float(balance_currency), 2)

      if balance_usd > MIN_BALANCE_USD:
        portfolio.append({
            'currency': account.currency,
            'balance_coin': balance_currency,
            'balance_usd': balance_usd
        })

  total_balance_usd = 0
  for holding in portfolio:
    total_balance_usd += holding['balance_usd']

  for holding in portfolio:
    holding['allocation'] = round(holding['balance_usd'] / total_balance_usd * 100, 2)

  return balance_usdc, portfolio

def portfolio_table(portfolio):
  table = PrettyTable(["Currency", "Balance (Coin)", "Balance (USD)", "Allocation"])
  table.align["Currency"] = "l"
  table.align["Balance (Coin)"] = "r"
  table.align["Balance (USD)"] = "r"
  table.align["Allocation"] = "r"
  for item in portfolio:
      table.add_row([item["currency"], item["balance_coin"], "{:.2f}".format(item["balance_usd"]), "{:.2f}%".format(item["allocation"])])
  return table

def predictions_table(predictions):
  table = PrettyTable(["Symbol", "# 4% Runs", "Time Above 4%"])
  table.align["Symbol"] = "l"
  table.align["# 4% Runs"] = "r"
  table.align["Time Above 4%"] = "r"
  for index, prediction in predictions.iterrows():
    table.add_row([prediction["Symbol"], prediction["Hit Count"], "{:.2f}%".format(prediction["Min Above"] * 100.0)])
  return table

def build_purchase_decision_prompt(predictions, portfolio, balance_usdc):
  prompt = ""
  prompt += "PREDICTIONS:\n"
  prompt += str(predictions_table(df_filtered)) + "\n"
  prompt +=  "Anything shown here is appropriate to buy.\n\n"
  prompt += "- Symbol: The Coinbase ticker symbol"
  prompt += "- # 4% Runs: How many times has this coin gone up 4% in the last 14 days. A signal the coin is volitile enough to cover the fees.\n"
  prompt += "- Time Above 4%: In the last two weeks, what percentage of time has the value been over 4%? Prefer values around 50%.\n"
  prompt += "\n"
  prompt += "CURRENT PORTFOLIO:\n"
  prompt += str(portfolio_table(portfolio)) + "\n\n"
  prompt += f"USD AVAILABLE TO INVEST: ${balance_usdc}"
  prompt += "\n"
  prompt += "You are part of an automated cryptocurrency trading system tasked with evaluating the best currency to buy.\n"
  prompt += "Select a coin from PREDICTIONS to purchase next. Try to:\n"
  prompt += "- Maintain a balanced portfolio, preferring new coins and avoiding allocation over 10%\n"
  prompt += "- Pick '# 4% Runs' with higher values when possible\n"
  prompt += "- Pick 'Time Above 4%' near 50% if possible\n"
  prompt += "\n"
  prompt += "Select ONE symbol for this purchase.\n"
  prompt += "Format your answer as: PURCHASE[SYMBOL]"
  return prompt

def parse_llm_purchase_response(response):
  pattern = r"PURCHASE\[(.*?)\]"
  match = re.search(pattern, response)
  if match:
    symbol = match.group(1)
  else:
    symbol = ""
  return symbol

def gptquery(prompt):
  # Set up the API endpoint URL
  url = 'https://api.openai.com/v1/chat/completions'

  # Set your OpenAI API key


  # Send a POST request to the API
  response = requests.post(
      url,
      headers={
          'Content-Type': 'application/json',
          'Authorization': f'Bearer {api_key}'
      },
      json={
          # 'prompt': prompt,
          "model": "gpt-4",
          "messages": [{"role": "user", "content": prompt}],
          # 'max_tokens': 50,  # Adjust the maximum number of tokens in the response
          'temperature': 0.6,  # Adjust the temperature for response randomness
          # 'n': 1,  # Adjust the number of responses to return
          # 'stop': None,  # Optional stop sequence to end the generated response
      }
  )


  data = response.json()
  # print(data)
  result = data['choices'][0]['message']['content']
  return result

def parse_llm_purchase_response(response):
  pattern = r"PURCHASE\[(.*?)\]"
  match = re.search(pattern, response)
  if match:
    symbol = match.group(1)
  else:
    symbol = ""
  return symbol

def build_client_order_id(symbol):
    DATE_FORMAT = "%Y%m%d%H%M%S"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(DATE_FORMAT)
    return f"BUY_{symbol}_{timestamp}"

def decimial_precision(value):
    # Convert the float to a string and split it at the decimal point.
    string_value = str(value)
    split_value = string_value.split(".")

    # If there is no decimal point, return 0.
    if len(split_value) == 1:
        return 0

    # Otherwise, return the length of the decimal part.
    else:
        return len(split_value[1])
    
def make_purchase_decision(predictions, portfolio, balance_usdc):
  prompt = build_purchase_decision_prompt(predictions, portfolio, balance_usdc)
  result = gptquery(prompt)
  return parse_llm_purchase_response(result)

purchase_symbol = parse_llm_purchase_response(llm_purchase_response)
product_id = f"{purchase_symbol}-USDC"


product = coinbaseClient.get_product(product_id)
base_decimals = decimial_precision(product.base_increment)
quote_decimals = decimial_precision(product.quote_increment)

ask_price = float(coinbaseClient.get_best_bid_ask([product_id]).pricebooks[0].asks[0].price)

highest_bid = "{:.{prec}f}".format(ask_price * 1.005, prec=quote_decimals)

order_id = build_client_order_id(purchase_symbol)
amount_usd = min(5.0, balance_usdc)

base_size = "{:.{prec}f}".format(amount_usd / ask_price, prec=base_decimals)

print("BUYING")
print(f"Symbol:       {purchase_symbol}")
print(f"Asking Price: {ask_price}")
print(f"Highest Bid:  {highest_bid}")
print(f"Amount USD:   {amount_usd}")
print(f"Base Size:    {base_size}")

# create_order = coinbaseClient.create_limit_order(
#     client_order_id=order_id,
#     product_id=product_id,
#     side=Side.BUY,
#     base_size=base_size,
#     limit_price=highest_bid)

MAX_RETRIES = 100

retry_count = 0
while coinbaseClient.get_order(create_order.order_id).status != 'FILLED':
  time.sleep(1)
  retry_count += 1
  if retry_count > MAX_RETRIES:
    print("Never filled!")
    break

order = coinbaseClient.get_order(create_order.order_id)

def tradebot_string(product_id)
print(f"OPEN,SELL,{product_id},{order.filled_value},{order.average_filled_price},0.96,1.1")

