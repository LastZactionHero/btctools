while true; do
    echo "$(date)"
    scp zach@144.202.24.235:/home/zach/btctools/crypto_exchange_rates.csv ./data/
    python3 ./scripts/analysis/multi_predict_103.py ./data/crypto_exchange_rates.csv ./data/lstm_predictions.csv
    scp ./data/lstm_predictions.csv zach@144.202.24.235:/home/zach/btctools/lstm_predictions.csv
    sleep 60
done