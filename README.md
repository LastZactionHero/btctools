TODOS:
- Fix SimulatedHoldings not making sale
- Replay simulator

IDEAS:
- Global Market Condition (good time to buy or dump?)


Data: 144.202.24.235
Magpie: 108.61.193.193

## Setup

``````
sudo apt update  # Update package list
sudo apt install sqlite3
```

Experiments:


- exp0: 
result: ~$910
hits: 20
misses: 29

01/23/24
LSTM Series
PRIOR_MINUTES_TO_REVIEW = 14 * 24 * 60  # 14 days
POSITIVE_PREDICTION_MIN_DELTA = 2.0
MIN_POSITIVE_PREDICTIONS = 0
MAX_BUY_AMOUNT_USDC = 100
MIN_BUY_AMOUNT_USDC = 50
STOP_LOSS_PERCENT = 0.98
PROFIT_PERCENT = 1.1
TIME_ABOVE_MAX_PERCENTAGE = 1.0
TIME_ABOVE_MIN_PERCENTAGE = 0.0
MIN_HIT_COUNT = 2
HIT_FACTOR = 1.04
PREDICTION_SEQUENCE_LOOKBEHIND_DAYS = 7

- exp1: 
01/23/24, 8:40 AM
raised stop-loss to 0.98
RAISE_STOPLOSS_THRESHOLD = 1.0d1