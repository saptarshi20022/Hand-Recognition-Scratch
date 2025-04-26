import cv2
import mediapipe as mp
# import time

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Updated for MediaPipe 0.10.3+ with current parameter names
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        """
        Returns:
        - img: Image with drawn landmarks
        - bboxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        bboxes = []

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate bounding box with margin (from Project 1)
                h, w, _ = img.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                margin = 20
                x_min = max(int(min(x_coords) - margin), 0)
                y_min = max(int(min(y_coords) - margin), 0)
                x_max = min(int(max(x_coords) + margin), w)
                y_max = min(int(max(y_coords) + margin), h)
                
                bboxes.append((x_min, y_min, x_max, y_max))

        return img, bboxes

    def find_position(self, img, hand_idx=0, draw=True, include_z=False):
        """
        Returns landmark positions for a specific hand
        Updated to match MediaPipe's current landmark numbering
        """
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_idx]
            
            for idx, lm in enumerate(hand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if include_z:
                    lm_list.append([idx, cx, cy, lm.z])
                else:
                    lm_list.append([idx, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return lm_list

# Example usage (keep for testing)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_num_hands=1)
    
    while True:
        success, img = cap.read()
        img, bboxes = detector.find_hands(img)
        
        if bboxes:
            # Drawing first hand's bounding box
            x_min, y_min, x_max, y_max = bboxes[0]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()