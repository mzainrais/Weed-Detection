import cv2
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
try:
    my_new_model = YOLO('./best.pt')
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit()

# Set confidence threshold
conf_threshold = 0.7

# Function to process a frame and get detections
def process_frame(frame, model, conf_threshold):
    results = model(frame)
    detections = []
    if results and results[0].boxes:  # Check if there are any detections
        for box in results[0].boxes:
            if box.conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
                detections.append((x1, y1, x2, y2, box.conf.item(), box.cls.item()))
    return detections

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use the correct index or path for your camera

if not cap.isOpened():
    print("Error: Camera not initialized")
    exit()

print("Camera initialized successfully")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get detections
    detections = process_frame(frame, my_new_model, conf_threshold)

    # Draw bounding boxes and labels on the frame
    for (x1, y1, x2, y2, conf, cls) in detections:
        label = f'Weed {cls}: {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Live Weed Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Resources released and script ended")