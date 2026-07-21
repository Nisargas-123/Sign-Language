import mysql.connector
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="nSp@2004",   # <-- change this
    database="sign_language"
)
cursor = db.cursor()

# Load data
cursor.execute("SELECT label, landmarks FROM sign_samples")
rows = cursor.fetchall()

data = []
labels = []

for label, landmarks_json in rows:
    data.append(json.loads(landmarks_json))
    labels.append(ord(label) - ord('A'))

data = np.array(data)
labels = np.array(labels)

print("Dataset Loaded:", data.shape, "samples")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save model
model.save("sign_language_model.h5")
print("Model saved as sign_language_model.h5")

cursor.close()
db.close()
