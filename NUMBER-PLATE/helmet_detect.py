import cv2
import os
from keras.models import load_model
import numpy as np

# Load your trained model for helmet detection

model = load_model('helmetmodel.h5')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform helmet detection on the entire frame
    gray = cv2.resize(gray, (24, 24))
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=3)
    count =0
    # Perform helmet prediction using your trained model
    prediction = model.predict(gray)
   
    if prediction[0][0] > 0.5:
        label = "Wearing Helmet"
        color = (0, 255, 0)
    else:
        label = "Not Wearing Helmet"
        color = (0, 0, 255)
    
    cv2.putText(frame, label, (10, 30), font, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
