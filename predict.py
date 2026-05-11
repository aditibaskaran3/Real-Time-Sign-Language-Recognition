import cv2
import pyttsx3
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter

# Load model
model = joblib.load("model/sign_language_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

last_spoken = ""

# Webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store recent predictions
prediction_history = []

while True:

    success, img = cap.read()

    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])[0]

            # Save predictions
            prediction_history.append(prediction)

            # Keep last 15 predictions
            if len(prediction_history) > 15:
                prediction_history.pop(0)

            # Most common prediction
            stable_prediction = Counter(prediction_history).most_common(1)[0][0]
            if stable_prediction != last_spoken:

               print(f"Speaking: {stable_prediction}")

               engine.stop()

               engine.say(stable_prediction)

               engine.runAndWait()

               last_spoken = stable_prediction

            cv2.putText(
                img,
                f"Gesture: {stable_prediction}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Sign Language Prediction", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()