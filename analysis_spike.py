import sys
import pandas as pd
import numpy as np

OUTPUT_FILENAME = "./data/rise_series_real.csv"

# Find currencies that:
# Rise by a minimum 5%
MINIMUM_RISE_PERCENTAGE = 0.06
# ... in less than 30 minutes
MAX_RISE_MINUTES = 120
# ... and does not fall by 1%
MAX_FALL_PERCENTAGE = 0.02
# ... for at least 10 minutes.
MINIMUM_HOLD_MINUTES = 10
# Length of time to capture prior to rise sequence
PREFETCH_MINUTES = 60

file_path = sys.argv[1]
data = pd.read_csv(file_path) # 1754 x 80

found = {}

for row_idx, row in data.iterrows():
    if row_idx < MAX_RISE_MINUTES:
        continue

    for col_idx, col in enumerate(data.columns[1:]):
        if col in ['kyber-network', 'guacamole']:
            continue

        if col in found.keys() and found[col][-1] > row_idx - MAX_RISE_MINUTES:
            continue

        current = row[col]
        
        # Make sure we're not in a valley- no point in the last 30 minutes was above 5%
        max_recent_value = max(data[col][row_idx - MAX_RISE_MINUTES : row_idx])
        if((max_recent_value - current) / current > MINIMUM_RISE_PERCENTAGE):
           next

        possible = np.array([])
        
        # Find every point in the next 30 minutes that rises above 5%
        rise_range = data[col][row_idx : row_idx + MAX_RISE_MINUTES]
        for rise_idx, rise in enumerate(rise_range):
            rise_percentage = (rise - current) / current
            if rise_percentage > MINIMUM_RISE_PERCENTAGE:
                possible = np.append(possible, rise_idx)

        # # For each of these points, does it maintain above 4% for 10 minutes
        if(len(possible) > 0):
            for rise_idx in possible:
                rise = data[col][row_idx + rise_idx]

                hold_range = data[col][row_idx + int(rise_idx) : row_idx + int(rise_idx) + MINIMUM_HOLD_MINUTES]
                hold_time_minimum = min(hold_range)
                fall_percentage = (hold_time_minimum - rise) / rise
                if(fall_percentage > 0 or -1 * fall_percentage < MAX_FALL_PERCENTAGE ):
                    print("Found one!: {} {}".format(col, row_idx))
                    if col in found.keys():
                        found[col].append(row_idx)
                    else:
                        found[col] = [row_idx]
                    break

out = []

for coin in found:
    for time in found[coin]:
        prefetch_sequence = data[coin][time - PREFETCH_MINUTES : time]
        # import pdb; pdb.set_trace()
        out.append(prefetch_sequence.to_numpy().round(5))

out = np.vstack(out)
pd.DataFrame(out).to_csv(OUTPUT_FILENAME, index=False)