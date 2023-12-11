import sys
import pandas as pd
import numpy as np



# Find currencies that:
# Rise by a minimum 5%
MINIMUM_RISE_PERCENTAGE = 0.03
# ... in less than 30 minutes
MAX_RISE_MINUTES = 30
# ... and does not fall by 1%
MAX_FALL_PERCENTAGE = 0.02
# ... for at least 10 minutes.
MINIMUM_HOLD_MINUTES = 10
# Length of time to capture prior to rise sequence
PREFETCH_MINUTES = 120
# Offset into the run
PRERUN_OFFSET = 5

file_path = sys.argv[1]
data = pd.read_csv(file_path) # 1754 x 80

output_filename = sys.argv[2]

found = {}

for row_idx, row in data.iterrows():
    if row_idx < max(MAX_RISE_MINUTES, PREFETCH_MINUTES) or (row_idx + PRERUN_OFFSET) >= len(data) :
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
            if rise_percentage > MINIMUM_RISE_PERCENTAGE and rise_percentage < 0.05:
                possible = np.append(possible, rise_idx)

        # For each of these points, DOES NOT maintain above 4% for 10 minutes
        if(len(possible) > 0):
            for rise_idx in possible:
                rise = data[col][row_idx + rise_idx]

                hold_range = data[col][row_idx + int(rise_idx) : row_idx + int(rise_idx) + MINIMUM_HOLD_MINUTES]
                hold_time_minimum = min(hold_range)
                fall_percentage = (hold_time_minimum - rise) / rise
                if(-1 * fall_percentage > MAX_FALL_PERCENTAGE ):
                    print("It's a trap!: {} {}".format(col, row_idx))
                    if col in found.keys():
                        found[col].append(row_idx)
                    else:
                        found[col] = [row_idx]
                    break

out = []
for coin in found:
    for time in found[coin]:
        prefetch_sequence = data[coin][time - PREFETCH_MINUTES + PRERUN_OFFSET: time + PRERUN_OFFSET]
        out.append(prefetch_sequence.to_numpy().round(5))
out = np.vstack(out)
pd.DataFrame(out).to_csv(output_filename, index=False)