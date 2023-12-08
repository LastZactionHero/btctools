scp zach@144.202.24.235:/home/zach/btctools/crypto_exchange_rates.csv ./data/
python scripts/analysis/analysis_spike_fake.py ./data/crypto_exchange_rates.csv ./data/rise_series_fake.csv
python scripts/analysis/analysis_spike.py ./data/crypto_exchange_rates.csv ./data/rise_series_real.csv
python scripts/analysis/analysis_spike_classifier.py