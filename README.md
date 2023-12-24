```
pip install -r requirements.txt
```

scp zach@144.202.24.235:/home/zach/btctools/crypto_exchange_rates.csv ./data/

TODO:


TRIALS:
- Put prefetch sequence slightly into the run
- Train deeper into the run
- LSTM
- Longer run- 1,4,8 hours

EMBEDDING:
- Train an embedding over 30 minute periods
- K-Means into N groups
- Evaluate the patterns: are there clear "profit" and "loss" patterns?
- What are the patterns leading up to the "profit/loss" series? e.g. 1,1,4,2,1,*8* 

Encoder and cluster by trend correlation?