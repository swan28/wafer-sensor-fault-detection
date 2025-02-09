# colab code

# %pip install pymongo

from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# uniform resource indentifier
uri = "mongodb+srv://username:<password>@cluster0.tybza.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name
DATABASE_NAME="FaultDetectionProject"
COLLECTION_NAME="waferfault"

# read the data as a dataframe
df=pd.read_csv(r"./notebooks/wafer_fault.csv")
df.head()

# Convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

