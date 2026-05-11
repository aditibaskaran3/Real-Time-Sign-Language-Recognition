import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import joblib

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Dataset path
dataset_path = "dataset"

# Data storage
X = []
y = []

# Gesture labels
labels = os.listdir(dataset_path)

print("Loading dataset...")

for label in labels:

    folder_path = os.path.join(dataset_path, label)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                X.append(landmarks)
                y.append(label)

print("Training model...")

# Train model
model = RandomForestClassifier()

model.fit(X, y)

# Save model
joblib.dump(model, "model/sign_language_model.pkl")

print("Model trained and saved successfully!")