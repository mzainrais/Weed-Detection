import cv2
import numpy as np
from keras.models import load_model

# Load the H5 model
model = load_model('trained-model.h5')

# Define a function to preprocess the frames
def preprocess_frame(frame):
    # Resize frame to the input size expected by the model (e.g., 224x224)
    frame_resized = cv2.resize(frame, (224, 224))
    # Normalize pixel values to the range [0, 1]
    frame_normalized = frame_resized / 255.0
    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(preprocessed_frame)
    # Assuming it's a classification model, get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-Time Prediction', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
