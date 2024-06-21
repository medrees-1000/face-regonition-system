import numpy as np
import cv2 as cv

# Loading haar and the name of the poeple 
haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ["bilal", "Mohammad"]
  
# Loading the yml and face_recognizer we trained 
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Get a reference to webcam #0 (the default one)
cap = cv.VideoCapture(0) 

while True:
    
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Turn picture gray  
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    # Grabbing regen of interest 
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        # Predict the face and with how much confidence 
        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Label = {people[label]} with a confidence of {confidence}")

        # when to name  a person unknown 
        if confidence > 45:
            name = people[label]
        else:
            name = "Unknown"

        # Using cv to put text on the picture to display name 
        cv.putText(frame, f"{name} {confidence:.2f}", (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # show the webcam
    cv.imshow('Detected Faces', frame)
    
    # how to stop the script 
    key = cv.waitKey(20)
    if key & 0xFF == ord('A'):
        break

cap.release()
cv.destroyAllWindows()