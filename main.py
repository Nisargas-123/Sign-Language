import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import string
import mysql.connector

# -------------------------------
# MySQL Connection
# -------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="nSp@2004",
    database="sign_language"
)
cursor = db.cursor()

# -------------------------------
# Load Model
# -------------------------------
model = tf.keras.models.load_model("sign_language_model.h5")
labels = list(string.ascii_uppercase)

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.4
last_prediction = None   # To store only when it changes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_label = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label
        hand_label = handedness

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract 63 landmark features
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            preds = model.predict(np.array(landmarks).reshape(1, 63))
            confidence = np.max(preds)

            if confidence > CONFIDENCE_THRESHOLD:
                predicted_label = labels[np.argmax(preds)]

                # Show on screen
                cv2.putText(frame, f"Letter: {predicted_label}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

                # --------------------------------------------
                # STORE IN MYSQL ONLY WHEN PREDICTION CHANGES
                # --------------------------------------------
                if predicted_label != last_prediction:
                    cursor.execute(
                        "INSERT INTO predictions (letter, confidence, handedness) VALUES (%s, %s, %s)",
                        (predicted_label, float(confidence), hand_label)
                    )
                    db.commit()

                    print(f"Saved prediction → {predicted_label}, conf={confidence:.2f}, hand={hand_label}")

                    last_prediction = predicted_label  # Update last prediction

    # Show Handedness
    if hand_label:
        cv2.putText(frame, hand_label, (frame.shape[1] - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
