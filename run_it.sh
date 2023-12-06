scp zach@144.202.24.235:/home/zach/btctools/crypto_exchange_rates.csv ./data/
python ./data_prep_price_to_delta.py
python ./data_prep_timestamp.py
python ./data_prep_smooth.py
python ./analysis_rnn.py