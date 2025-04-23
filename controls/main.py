import cv2
import time
import sys
import os
# import math
import numpy as np
import tensorflow as tf
import pyautogui
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hand_tracker import HandDetector

# Constants
wCam, hCam = 640, 480
confidence_threshold = 0.8

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Load gesture recognition model
model = tf.keras.models.load_model("gesture_recognition_model.keras")
gesture_labels = {
    0: "thumbs_up",
    1: "palm_open",
    2: "swipe_right", 
    3: "no_gesture"
}

# Initialize hand detector
detector = HandDetector(max_num_hands=1, min_detection_confidence=0.7)

# Control parameters
pTime = 0
screen_w, screen_h = pyautogui.size()
smoothening = 3
plocX, plocY = 0, 0
clocX, clocY = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Detect hands and get bounding boxes
    img, bboxes = detector.find_hands(img)
    lmList = detector.find_position(img, include_z=True)

    current_gesture = "no_gesture"
    
    if bboxes and lmList:
        # Get first hand's bounding box
        x_min, y_min, x_max, y_max = bboxes[0]
        
        # Crop and preprocess hand region
        hand_roi = img[y_min:y_max, x_min:x_max]
        if hand_roi.size > 0:
            resized = cv2.resize(hand_roi, (64, 64))
            normalized = resized / 255.0
            prediction = model.predict(normalized[np.newaxis, ...], verbose=0)[0]
            
            if np.max(prediction) > confidence_threshold:
                current_gesture = gesture_labels[np.argmax(prediction)]

    # Gesture-Action Mapping
    if current_gesture == "palm_open" and lmList:
        # Cursor Control Logic
        x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
        
        # Convert coordinates to screen size
        x3 = np.interp(x1, (0, wCam), (0, screen_w))
        y3 = np.interp(y1, (0, hCam), (0, screen_h))
        
        # Smoothen movement
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
        
        pyautogui.moveTo(clocX, clocY)
        plocX, plocY = clocX, clocY

    elif current_gesture == "thumbs_up":
        pyautogui.click()
        time.sleep(0.3)  # Debounce click

    elif current_gesture == "swipe_right":
        #pyautogui.hscroll(100)
        # Fallback Method 2: Simulate key press
        pyautogui.press('right')
        time.sleep(0.3)

    # FPS Display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Gesture Status
    cv2.putText(img, f'Mode: {current_gesture}', (10, 100),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    cv2.namedWindow("Gesture Control")
    cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("Gesture Control", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()