import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define gesture label and output directory
gesture_label = "thumbs_up"  # Change for different gestures
output_dir = f"dataset/{gesture_label}"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving images to: {output_dir}")

cap = cv2.VideoCapture(0)
image_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    # Draw hand landmarks and save the cropped ROI
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Expand the bounding box
            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            # Crop and save the hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size > 0:
                file_path = os.path.join(output_dir, f"{image_id}.jpg")
                cv2.imwrite(file_path, hand_roi)
                image_id += 1
                print(f"Image saved: {file_path}")

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
