import numpy as np
import cv2 as cv

# Loading haar cascade and the names of the people
haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ["Mohammad", "Marcos Pinto", "Sarah", "Nazia Mahmud"]

# Loading the trained face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Get a reference to webcam #0 (the default one)
cap = cv.VideoCapture(0)

while True:
    
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    
    # Process each detected face
    for (x, y, w, h) in faces_rect:
        # Extract the region of interest (ROI) which is the face
        faces_roi = gray[y:y+h, x:x+w]

        # Predict the face label and confidence level
        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Label = {label} with a confidence of {confidence}")

        # Determine the name to display
        if 0 <= label < len(people):
            name = people[label]
        else:
            name = "Unknown"

        # Display the name and confidence level on the frame
        print(f"Displaying name: {name} with confidence {confidence:.2f}")
        if name == "Unknown":
            cv.putText(frame, f"{name}", (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv.putText(frame, f"{name} {confidence:.2f}", (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the frame with detected faces
    cv.imshow('Recognize The Poeple', frame)
    
    # Check for user input to stop the script (press 'A' key to exit)
    key = cv.waitKey(1)
    if key == ord('A'):
        break

# Release the capture and close all windows
cap.release()
cv.destroyAllWindows()
