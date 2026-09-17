# Importing Libraries
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Download the model once if not present in the project folder.
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initializing the model using MediaPipe Tasks API.
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.75,
    min_tracking_confidence=0.75,
    running_mode=vision.RunningMode.VIDEO,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)
frame_index = 0

while True:
    # Read video frame by frame
    success, img = cap.read()
    if not success:
        break

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Process the RGB image
    frame_index += 1
    timestamp_ms = frame_index * 33
    results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    # If hands are present in image(frame)
    if results.handedness:
        labels = [h[0].category_name for h in results.handedness]

        # Both Hands are present in image(frame)
        if len(labels) == 2:
            cv2.putText(
                img,
                "Both Hands",
                (250, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        else:
            label = labels[0]
            if label == "Right":
                cv2.putText(
                    img,
                    label + " Hand",
                    (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            if label == "Left":
                cv2.putText(
                    img,
                    label + " Hand",
                    (460, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

    # Display Video and when 'q' is entered, destroy the window
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()