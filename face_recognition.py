import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ["bilal", "Mohammad"]
  

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Label = {people[label]} with a confidence of {confidence}")

        if confidence > 45:
            name = people[label]
        else:
            name = "Unknown"

        cv.putText(frame, f"{name} {confidence:.2f}", (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('Detected Faces', frame)
    key = cv.waitKey(20)
    if key & 0xFF == ord('A'):
        break

cap.release()
cv.destroyAllWindows()