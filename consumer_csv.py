from kafka import KafkaConsumer
import json
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.fs as fs
import pandas as pd

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'fraud_predictions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# HDFS configuration
hdfs_host = 'localhost'
hdfs_port = 9000
hdfs_path = '/user/hadoop/predictions.csv'

# Connect to HDFS
hdfs_filesystem = fs.HadoopFileSystem(host=hdfs_host, port=hdfs_port)

# Check if the file already exists on HDFS and load it
if hdfs_filesystem.exists(hdfs_path):
    with hdfs_filesystem.open_input_file(hdfs_path) as f:
        table = csv.read_csv(f)
    data = table.to_pandas()  # Convert to pandas DataFrame
else:
    data = pd.DataFrame(columns=['is_fraud', 'predicted_value'])  # Start with empty DataFrame

# Listen for Kafka messages and append them to DataFrame
for message in consumer:
    record = message.value
    new_row = pd.DataFrame([record])  # Create a DataFrame for the new row
    data = pd.concat([data, new_row], ignore_index=True)  # Append new row to the DataFrame

    # Write the updated DataFrame to HDFS
    arrow_table = pa.Table.from_pandas(data)

    try:
        with hdfs_filesystem.open_output_stream(hdfs_path) as f:
            csv.write_csv(arrow_table, f)
        print(f"Appended row to HDFS CSV: {record}")
    except Exception as e:
        print(f"Failed to write to HDFS: {e}")
