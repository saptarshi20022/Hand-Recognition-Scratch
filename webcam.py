import cv2

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored effect
    frame = cv2.flip(frame, 1)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()