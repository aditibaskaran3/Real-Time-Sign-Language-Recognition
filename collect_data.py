import cv2
import os
import time

gesture_name = "help"

dataset_path = f"dataset/{gesture_name}"

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = len(os.listdir(dataset_path))

while count < 100:

    success, img = cap.read()

    if not success:
        break

    cv2.putText(
        img,
        f"Collecting Images: {count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Collection", img)

    image_path = f"{dataset_path}/{count}.jpg"

    cv2.imwrite(image_path, img)

    print(f"Saved: {image_path}")

    count += 1

    time.sleep(0.2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()