scp zach@144.202.24.235:/home/zach/btctools/crypto_exchange_rates.csv ./data/
python ./analysis_spike_fake.py ./data/crypto_exchange_rates.csv
python ./analysis_spike.py ./data/crypto_exchange_rates.csv     
python ./analysis_spike_classifier.py