import cv2
from ultralytics import YOLO

# Use your trained model OR a base model for testing
# model = YOLO("models/cash_yolov8.pt")  # for trained cash detector
model = YOLO("yolov8n.pt")  # test only (won't detect cash unless trained)

# Open laptop camera (usually index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run detection on the frame
    results = model(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Show it in a window
    cv2.imshow("Live Cash Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
