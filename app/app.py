import streamlit as st
import cv2
import mediapipe as mp
import joblib
from collections import Counter
# Load trained model
model = joblib.load("../model/sign_language_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Streamlit UI
st.title("🧏 Sign Language Recognition System")
st.write("Real-time gesture recognition using ML and Computer Vision")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# Webcam
cap = cv2.VideoCapture(0)

prediction_history = []

while run:

    success, frame = cap.read()

    if not success:
        st.write("Failed to access webcam")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])[0]

            prediction_history.append(prediction)

            if len(prediction_history) > 15:
                prediction_history.pop(0)

            stable_prediction = Counter(prediction_history).most_common(1)[0][0]

            cv2.putText(
                frame,
                f"Gesture: {stable_prediction}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()