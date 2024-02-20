TODOS:
- Fix SimulatedHoldings not making sale
- Replay simulator

IDEAS:
- Global Market Condition (good time to buy or dump?)
- Auto-lower stop loss 

Data: 144.202.24.235
Magpie: 45.63.61.60

## Setup

``````
sudo apt update  # Update package list
sudo apt install sqlite3
```

LIVE PREP:
- Remove debugs
- Logger

TODOS:
- Ship it!


sweeps
| Parameter                   | Status      | Result   |
|-----------------------------|-------------|----------|
| buy_interval_minutes        | Not Started |          |
| raise_stoploss_threshold    | Complete    |  1.018   |
| sell_stoploss_floor         | Complete    |  0.00184 |
| stop_loss_percent           | Complete    |  0.07820 |
| max_delta                   | Complete    |  4.30120 |
| max_spread                  | Complete    |  None    |
| sell_all_on_hit             | Complete    |  No      |
| loss_recovery_after_minutes | Complete    |  4d      |
| single_buy                  | Complete    |  True    |


| Category      | Avg hold time|Median hold time|Max hold time|Min hold time|Std Dev hold time|
| ------------- |-------------:| --------------:|------------:|------------:|----------------:|
| SOLD orders   |1.55 days     |     1.79 days |2.67 days    |0.00 days    |0.90 days        |
| OPEN orders   |6.89 days     |     7.43 days |8.50 days    |5.32 days    |1.05 days        |
| Profit orders |1.55 days     |     1.79 days |2.67 days    |0.00 days    |0.90 days        |

