from kafka import KafkaConsumer
import json
import pandas as pd
import os

# Set up Kafka consumer
consumer = KafkaConsumer(
    "fraud_predictions",
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud_detection_group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Local CSV file path
local_csv_path = 'fraud_predictions.csv'

# Create the CSV file and write the header if it doesn't exist
if not os.path.exists(local_csv_path):
    # Create an empty DataFrame with the desired columns
    initial_data = pd.DataFrame(columns=['is_fraud', 'predicted_value'])  # Adjust columns as per your data structure
    initial_data.to_csv(local_csv_path, index=False)

print("Listening for fraud prediction messages...")

for message in consumer:
    prediction = message.value
    print(f"Received prediction: {prediction}")

    # Append the received prediction to the CSV file
    new_row = pd.DataFrame([prediction])  # Create a DataFrame for the new row
    new_row.to_csv(local_csv_path, mode='a', header=False, index=False)  # Append without header

    print(f"Appended prediction to CSV: {prediction}")

print("Finished processing messages.")
