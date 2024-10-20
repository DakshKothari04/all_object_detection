import cv2
import os
import tensorflow
from keras.models import load_model
from keras.metrics import MeanSquaredError
import numpy as np

# Load your trained model for helmet detection
model_path='C:/others/tripling_detection/triplingmodel.h5'
model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (24, 24))

    # Preprocess for prediction
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=3)

    # Perform prediction using your trained model
    prediction = model.predict(gray)

    # Determine label and color based on prediction
    if prediction[0][0] > 0.5:
        label = "not tripling"
        color = (0, 255, 0)  
    else:
        label = "tripling"
        color = (0, 0, 255) 

        # Extract bounding box from prediction (if available from your model)
        # Assuming your model outputs bounding box coordinates (x_min, y_min, x_max, y_max) along with probability
    if len(prediction[0]) == 5:  # Check if prediction has bounding box info
        x_min = int(prediction[0][1] * width)
        y_min = int(prediction[0][2] * height)
        x_max = int(prediction[0][3] * width)
        y_max = int(prediction[0][4] * height)
        
        
        
        if prediction[0][0] > 0.5:  # Adjust threshold as needed
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Display label and bounding box (if applicable)
    cv2.putText(frame, label, (10, 30), font, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
