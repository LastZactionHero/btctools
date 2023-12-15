scp zach@144.202.24.235:/home/zach/btctools/crypto_exchange_rates.csv ./data/
python ./scripts/data_prep/data_prep_price_to_delta.py ./data/crypto_exchange_rates.csv ./data/delta.csv
python ./scripts/analysis/encoder_full_list.py ./data/delta.csv