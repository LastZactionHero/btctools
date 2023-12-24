# Import necessary libraries
import sys
import pandas as pd
import numpy as np

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(sys.argv[1], index_col=0)

lags = [45, 60, 75, 90, 105, 120]

correlations = []

for lag_idx, lag in enumerate(lags):
    print("{} minutes ({}/{})".format(lag, lag_idx, len(lags)))
    for leading_idx, leading_col in enumerate(df.columns):
        # print("{} ({}/{})".format(leading_col, leading_idx, len(df.columns)))
        leading_data = df[leading_col].values[:-lag]

        for trailing_idx, trailing_col in enumerate(df.columns):
            if leading_idx == trailing_idx:
                continue

            trailing_data = df[trailing_col].values[lag:]

            correlation = np.corrcoef(leading_data, trailing_data)[0,1]
            correlations.append({
                "leading": leading_col,
                "trailing": trailing_col,
                "correlation": correlation,
                "lag": lag
            })


correlations = sorted(correlations, key=lambda x: x['correlation'], reverse=True)

print("Positive")
for i in range(50):
    correlation = correlations[i]
    print("{}\t{}\t{}\t{}".format(correlation['correlation'].round(3), correlation['lag'], correlation['leading'], correlation['trailing']))

print("\n\n")

print("Negative")
for i in range(50):
    correlation = correlations[-i-1]
    print("{}\t{}\t{}\t{}".format(correlation['correlation'].round(3), correlation['lag'], correlation['leading'], correlation['trailing']))