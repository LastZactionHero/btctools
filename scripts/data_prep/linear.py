import pandas as pd
import numpy as np
import sys

# Load your data
file_path = sys.argv[1]
data = pd.read_csv(file_path)

l_range = 30
increment = 15

timestamps = data['timestamp']
prices = data['bitcoin']

linear_points = np.array([[0, prices[0]]])

current_time = 0
last_iteration = None

while True:
    price_range = prices[current_time:(current_time+l_range)]
    range_time_min = None
    range_time_max = None
    range_price_min = None
    range_price_max = None

    for range_idx, price in enumerate(price_range):
        if range_price_min is None or price < range_price_min:
            range_price_min = price
            range_time_min = range_idx + current_time
        if range_price_max is None or price > range_price_max:
            range_price_max = price
            range_time_max = range_idx + current_time

    if(range_time_min < range_time_max):
        if last_iteration == "positive":
            linear_points = linear_points[:-1]
            linear_points = np.vstack((linear_points, np.array([[range_time_max, range_price_max]])))
        else:
            linear_points = np.vstack((linear_points, np.array([[range_time_min, range_price_min]])))
            linear_points = np.vstack((linear_points, np.array([[range_time_max, range_price_max]])))
        last_iteration = "positive"
    else:
        if last_iteration == "negative":
            linear_points = linear_points[:-1]
            linear_points = np.vstack((linear_points, np.array([[range_time_min, range_price_min]])))
        else:
            linear_points = np.vstack((linear_points, np.array([[range_time_max, range_price_max]])))
            linear_points = np.vstack((linear_points, np.array([[range_time_min, range_price_min]])))
        last_iteration = "negative"

    current_time += increment
    if (current_time + increment > len(prices)):
        break

linearized = np.array([linear_points[0]])

for point in linear_points[1:]:
    last_point = linearized[-1]
    if point[0] == last_point[0]:
        continue

    run = point[0] - last_point[0]
    rise = point[1] - last_point[1]
    slope = float(rise) / float(run)

    # print("Run: {} Rise: {} Slope: {}".format(run, rise, slope))
    for i in range(1,run-1):
        linearized = np.vstack((linearized, np.array([[int(last_point[0] + i + 1), int(last_point[1] + slope * i)]])))
    linearized = np.vstack((linearized, point))

print(linearized)
# print(len(linearized))
# import pdb; pdb.set_trace()

print(len(linearized) + len(linear_points) / 2)
print(len(prices) )