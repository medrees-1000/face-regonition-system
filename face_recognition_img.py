import numpy as np 
import cv2 as cv


haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ["Bilal", "Mohammad"]

# features = np.load('features.npy')
# labels = np.load('labesl.npy')


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


img_path = 'C:\\Users\\idree\\Desktop\\mix of my stuff\\20200426_224234.JPG'
img = cv.imread(img_path)


if img is None:
    print(f"Error: Unable to load image at {img_path}")
else:
    print("Image loaded successfully")

    resize = cv.resize(img, (700, 500), interpolation=cv.INTER_LINEAR)
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    #cv.imshow('Person', gray)

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 6)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)

        if confidence < 100:  # Adjust this threshold as needed
            name = people[label]
        else:
            name = "Unknown"
        
        print(f"Label = {name} with a confidence of {confidence}")

        # Adjust the coordinates for the text
        cv.putText(resize, f"{name} {confidence:.2f}", (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        cv.rectangle(resize, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Detected Faces', resize)
    cv.waitKey(0)
    cv.destroyAllWindows()