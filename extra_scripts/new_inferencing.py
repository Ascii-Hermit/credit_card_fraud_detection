import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexerModel, VectorAssembler, StandardScalerModel
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.sql.functions import col, log
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName('FraudDetectionInference') \
    .master("local[*]") \
    .getOrCreate()

# Load the saved models
model_path = "fraud_detection_model"
model = DecisionTreeClassificationModel.load(model_path)
type_indexer = StringIndexerModel.load("models/type_indexer")
nameDest_indexer = StringIndexerModel.load("models/nameDest_indexer")
assembler = VectorAssembler.load("models/assembler")
scaler = StandardScalerModel.load("models/scaler")
print("Models loaded.")

# Get new instance input
type_input = input("Enter transaction type (e.g., CASH_OUT, TRANSFER): ")
amount = float(input("Enter transaction amount: "))
oldbalanceDest = float(input("Enter old balance for destination account: "))
newbalanceDest = float(input("Enter new balance for destination account: "))
oldbalanceOrg = float(input("Enter old balance for originating account: "))
newbalanceOrig = float(input("Enter new balance for originating account: "))
nameDest_input = input("Enter destination name: ")

# Create DataFrame for the new instance
new_instance = pd.DataFrame({
    "type": [type_input],
    "amount": [amount],
    "oldbalanceDest": [oldbalanceDest],
    "newbalanceDest": [newbalanceDest],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig],
    "nameDest": [nameDest_input]
})
new_instance = spark.createDataFrame(new_instance)

# Apply transformations
new_instance = type_indexer.transform(new_instance)
new_instance = nameDest_indexer.transform(new_instance)
new_instance = new_instance.withColumn("diffDist", col("oldbalanceDest") - col("newbalanceDest"))
new_instance = new_instance.withColumn("diffOrg", col("oldbalanceOrg") - col("newbalanceOrig"))

# Adjust columns for log transformation
for col_name in ['amount', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig']:
    min_value = new_instance.agg({col_name: "min"}).collect()[0][0]
    shift_value = abs(min_value) + 1 if min_value <= 0 else 0
    new_instance = new_instance.withColumn(col_name, log(col(col_name) + shift_value))

# Apply vector assembly and scaling
new_instance = assembler.transform(new_instance)
new_instance = scaler.transform(new_instance)

# Perform prediction
prediction = model.transform(new_instance)
prediction.show()

# Stop Spark session
spark.stop()
