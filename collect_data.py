import cv2
import numpy as np
import mediapipe as mp
import time
import mysql.connector
import json
import string

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="nSp@2004",   # Your MySQL password
    database="sign_language"
)
cursor = db.cursor()

# Constants
SAMPLES_PER_CLASS = 200
WARNING_DISPLAY_TIME = 3

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

letters = list(string.ascii_uppercase)
current_letter = None
current_number = 0   # For number-based selection

# ROI
roi_top_left = (200, 100)
roi_bottom_right = (450, 400)

def is_hand_inside_roi(landmarks, tl, br, frame_shape):
    h, w, _ = frame_shape
    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if not (tl[0] <= x <= br[0] and tl[1] <= y <= br[1]):
            return False
    return True

# Count samples already in DB
cursor.execute("SELECT label, COUNT(*) FROM sign_samples GROUP BY label")
db_counts = dict(cursor.fetchall())
sample_count = {l: db_counts.get(l, 0) for l in letters}

warning_message = None
warning_start_time = None

print("---------------------------------------------------------")
print("NUMBER SELECTION MODE ENABLED")
print("1 → A, 2 → B, ..., 26 → Z")
print("Press digits to form a number (Example: 1 then 0 → 10 = J)")
print("Press 's' to save sample.")
print("Press ESC to quit.")
print("---------------------------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)

    hand_present = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_present = True

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show warning message
    if warning_message:
        elapsed = time.time() - warning_start_time
        if elapsed < WARNING_DISPLAY_TIME:
            cv2.putText(frame, warning_message, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            warning_message = None

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == 27:
        print("Exiting.")
        break

    # -----------------------------
    # NUMBER-BASED LETTER SELECTION
    # -----------------------------
    if key in range(ord('0'), ord('9') + 1):
        digit = int(chr(key))

        # Build number: e.g., pressing 1 then 0 becomes 10
        current_number = current_number * 10 + digit
        print(f"Number entered: {current_number}")

        # Valid number → choose letter
        if 1 <= current_number <= 26:
            current_letter = chr(ord('A') + current_number - 1)
            print(f"Selected letter: {current_letter}")

        # Invalid → Reset
        if current_number > 26:
            print("Invalid number! Enter 1–26 only.")
            current_number = 0  # Reset

        continue

    # SAVE SAMPLE
    elif key == ord('s'):
        if current_letter is None:
            print("Select a letter using numbers first.")
            continue

        if not hand_present:
            warning_message = "No hand detected."
            warning_start_time = time.time()
            continue

        if not is_hand_inside_roi(hand_landmarks, roi_top_left, roi_bottom_right, frame.shape):
            warning_message = "Hand must be inside the green box."
            warning_start_time = time.time()
            continue

        if sample_count[current_letter] >= SAMPLES_PER_CLASS:
            print(f"{current_letter} already has {SAMPLES_PER_CLASS} samples.")
            continue

        # Extract 63 landmark values
        landmarks = [coord
                     for lm in hand_landmarks.landmark
                     for coord in (lm.x, lm.y, lm.z)]

        # Insert into MySQL
        cursor.execute(
            "INSERT INTO sign_samples (label, landmarks) VALUES (%s, %s)",
            (current_letter, json.dumps(landmarks))
        )
        db.commit()

        sample_count[current_letter] += 1
        print(f"Saved sample {sample_count[current_letter]} for {current_letter}")

cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
