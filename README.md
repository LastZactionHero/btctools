TODOS:
- Fix SimulatedHoldings not making sale
- Replay simulator

IDEAS:
- Global Market Condition (good time to buy or dump?)
- Auto-lower stop loss 

Data: 144.202.24.235
Magpie: 108.61.193.193

## Setup

``````
sudo apt update  # Update package list
sudo apt install sqlite3
```

try: 
-sell all coin orders on sale

sweeps
| Parameter                   | Status      | Result |
|-----------------------------|-------------|--------|
| buy_interval_minutes        | Not Started |        |
| raise_stoploss_threshold    | Complete    |  1.018 |
| sell_stoploss_floor         | Complete    |  0.002 |
| stop_loss_percent           | Complete    |  0.08  |
| max_delta                   | Complete    |  4.5   |
| max_spread                  | Complete    |  None  |
| recovery_stoploss_threshold | Not Impl    |        |
| sell_all_on_hit             | In Progress |        |

Experiments:

magpie_sim_01.db

RAISE_STOPLOSS_THRESHOLD = 1.012
SELL_STOPLOSS_FLOOR = 0.005
ORDER_AMOUNT_USD = 100.0
STOP_LOSS_PERCENT = 0.96
TAKE_PROFIT_PERCENT = 1.1
MAX_DELTA = 2.0
MODEL_FILENAME = "./models/lstm_series_240m.h5"
SEQUENCE_LOOKBEHIND_MINUTES = 240
PREDICTION_LOOKAHEAD_MINUTES = 30

sim 02:
Experiment: Reduced STOP_LOSS_PERCENT to 0.92

RAISE_STOPLOSS_THRESHOLD = 1.012
SELL_STOPLOSS_FLOOR = 0.005
ORDER_AMOUNT_USD = 100.0
STOP_LOSS_PERCENT = 0.92
TAKE_PROFIT_PERCENT = 1.1
MAX_DELTA = 2.0
MODEL_FILENAME = "./models/lstm_series_240m.h5"
SEQUENCE_LOOKBEHIND_MINUTES = 240
PREDICTION_LOOKAHEAD_MINUTES = 30

sim 03:
Experiment:  STOP_LOSS_PERCENT to 0.95, SELL_STOPLOSS_FLOOR higher to 0.01 to capitalize on bigger potential gains

RAISE_STOPLOSS_THRESHOLD = 1.012
SELL_STOPLOSS_FLOOR = 0.01
ORDER_AMOUNT_USD = 100.0
STOP_LOSS_PERCENT = 0.95
TAKE_PROFIT_PERCENT = 1.1
MAX_DELTA = 2.0
MODEL_FILENAME = "./models/lstm_series_240m.h5"
SEQUENCE_LOOKBEHIND_MINUTES = 240
PREDICTION_LOOKAHEAD_MINUTES = 30

sim 04:
Experiment: Raises STOP_LOSS_PERCENT to middle 0.94, SELL_STOPLOSS_FLOOR to middle 0.0075

RAISE_STOPLOSS_THRESHOLD = 1.012
SELL_STOPLOSS_FLOOR = 0.005
ORDER_AMOUNT_USD = 100.0
STOP_LOSS_PERCENT = 0.94
TAKE_PROFIT_PERCENT = 1.1
MAX_DELTA = 2.0
MODEL_FILENAME = "./models/lstm_series_240m.h5"
SEQUENCE_LOOKBEHIND_MINUTES = 240
PREDICTION_LOOKAHEAD_MINUTES = 30

sim 05:
Experiment: STOP_LOSS_PERCENT to 0.93, spread max to 0.2%, max delta to 3.0

Kind of a mess...

MAX_SPREAD = 0.001
RAISE_STOPLOSS_THRESHOLD = 1.012
SELL_STOPLOSS_FLOOR = 0.005
ORDER_AMOUNT_USD = 100.0
STOP_LOSS_PERCENT = 0.93
TAKE_PROFIT_PERCENT = 1.1
MAX_DELTA = 3.0
MODEL_FILENAME = "./models/lstm_series_240m.h5"
SEQUENCE_LOOKBEHIND_MINUTES = 240
PREDICTION_LOOKAHEAD_MINUTES = 30