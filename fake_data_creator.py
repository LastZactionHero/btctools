import random
import time
import csv

def generate_fake_data(num_records=500):
    fake_data = []

    current_timestamp = int(time.time())  # Get the current timestamp

    # Generate data for approximately 50 lines
    for _ in range(num_records):
        # Increment the timestamp by 60 seconds for each record
        timestamp = current_timestamp + (_ * 60)

        # Generate a random Bitcoin price (adjust as needed)
        price_usd = random.uniform(40000, 42000)

        fake_data.append((timestamp, price_usd))

    return fake_data

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "price_usd"])  # Write header
        writer.writerows(data)

if __name__ == "__main__":
    fake_data = generate_fake_data()
    save_to_csv(fake_data, "fake_training_data.csv")
