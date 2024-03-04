import numpy as np
import pandas as pd
from scripts.db.models import init_db_engine, Order
from sqlalchemy.orm import sessionmaker
from scripts.trade.coingecko_coinbase_pairs import gecko_coinbase_currency_map

REVIEW_PRIOR_MINUTES = 30 * 24 * 60 # 30 days

def calc_hit_percentage(data, order):
    coingecko_id = product_to_coingecko(order.coinbase_product_id)
    timestamp = int(order.created_at.timestamp())
    purchase_price = order.purchase_price

    # Find row index with nearest timestamp
    nearest_idx = (data['timestamp'] - timestamp).abs().idxmin()

    prior_prices = data[coingecko_id][(nearest_idx - REVIEW_PRIOR_MINUTES):nearest_idx][data[coingecko_id] > purchase_price]
    return 100 * float(len(prior_prices)) / REVIEW_PRIOR_MINUTES


coinbase_gecko_currency_map = {value: key for key, value in gecko_coinbase_currency_map.items()}

def product_to_coingecko(product):
    return coinbase_gecko_currency_map[product.split("-")[0]]
    

FILENAME_CRYPTO_EXCHANGE_RATES = "./data/crypto_exchange_rates.csv"
data_crypto_exchange_rates = pd.read_csv(FILENAME_CRYPTO_EXCHANGE_RATES)

Order, sessionmaker, init_db_engine
# Load DB
engine = init_db_engine("./db/live_20240226.db")

ignore_open_ids = [61, 62, 63, 64, 65, 109, 112]


Session = sessionmaker(bind=engine)
session = Session()

orders_hits = session.query(Order).filter(Order.status == "SOLD").all()
orders_open = session.query(Order).filter(Order.status == "OPEN", ~Order.id.in_(ignore_open_ids)).all()

hits = []
misses = []

for order in orders_hits:
    hits.append(calc_hit_percentage(data_crypto_exchange_rates, order))

for order in orders_open:
    misses.append(calc_hit_percentage(data_crypto_exchange_rates, order))

hits = np.array(hits)
misses = np.array(misses)
hits_mean = np.mean(hits)
hits_median = np.median(hits)
hits_std = np.std(hits)
hits_percentiles = np.percentile(hits, [25, 50, 75])

misses_mean = np.mean(misses)
misses_median = np.median(misses)
misses_std = np.std(misses)
misses_percentiles = np.percentile(misses, [25, 50, 75])

print("Hits:")
print("Mean:", hits_mean)
print("Median:", hits_median)
print("Standard Deviation:", hits_std)
print("Minimum:", np.min(hits))
print("Maximum:", np.max(hits))
print("25th Percentile:", hits_percentiles[0])
print("50th Percentile (Median):", hits_percentiles[1])
print("75th Percentile:", hits_percentiles[2])

print("\nMisses:")
print("Mean:", misses_mean)
print("Median:", misses_median)
print("Standard Deviation:", misses_std)
print("Minimum:", np.min(misses))
print("Maximum:", np.max(misses))
print("25th Percentile:", misses_percentiles[0])
print("50th Percentile (Median):", misses_percentiles[1])
print("75th Percentile:", misses_percentiles[2])

session.close()