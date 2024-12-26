import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load the trained TensorFlow model
model = tf.keras.models.load_model("gesture_recognition_model.keras")
print("Model loaded successfully.")

# Define gesture labels
gesture_labels = {0: "thumbs_up", 1: "palm_open", 2: "swipe_right", 3: "no_gesture"}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Draw the hand landmarks and extract ROI if a hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box for the hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Expand the bounding box slightly for better ROI
            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            # Extract and preprocess the ROI
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:  # Ensure ROI is valid
                resized_roi = cv2.resize(roi, (64, 64))
                normalized_roi = resized_roi / 255.0

                # Predict gesture
                prediction = model.predict(np.expand_dims(normalized_roi, axis=0), verbose=0)
                confidence = np.max(prediction)
                gesture_index = np.argmax(prediction)

                # Filter predictions with low confidence
                if confidence > 0.6:
                    gesture = gesture_labels.get(gesture_index, "Unknown")
                else:
                    gesture = "no_gesture"

                # Display the gesture on the frame
                cv2.putText(frame, f"{gesture} ({confidence:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
