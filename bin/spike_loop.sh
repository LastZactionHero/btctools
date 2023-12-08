while true; do
    echo "$(date)"
    scp zach@144.202.24.235:/home/zach/btctools/crypto_exchange_rates.csv ./data/
    python ./analysis_spike_current.py ./models/spike.h5 ./data/crypto_exchange_rates.csv
    sleep 60
done