import mysql.connector
from pymongo import MongoClient
import json

# Connect to MySQL
mysql_db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="nSp@2004",
    database="sign_language"
)
mysql_cursor = mysql_db.cursor()

# Connect to MongoDB
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_db = mongo_client["sign_language"]
mongo_samples = mongo_db["samples"]

# Fetch data from MySQL
mysql_cursor.execute("SELECT label, landmarks FROM sign_samples")
rows = mysql_cursor.fetchall()

count = 0
for label, landmarks in rows:
    landmarks = json.loads(landmarks)  # Convert JSON string → list
    mongo_samples.insert_one({
        "label": label,
        "landmarks": landmarks
    })
    count += 1

print(f"MIGRATION COMPLETE: {count} samples moved from MySQL → MongoDB")
