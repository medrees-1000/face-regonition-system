import os 
import cv2 as cv
import numpy as np

# Load the name of the folder that you have and the poeples name with the Path
people = ["Mohammad", "Bilal","Marcos Pinto", "Sarah", "Nazia Mahmud" ]
DIR = r"C:\Downloads\people"    

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Initialize some variables
features = []
labels = []


def create_train():
    
    # Loop over each file 
    for person in people:
        path = os.path.join(DIR, person)
        # join the Path to the label 
        label = people.index(person)
        
        
        # Go over each picture in the folder 
        for img in os.listdir(path):
            
            # join the path and the picture 
            img_path = os.path.join(path, img)
            
            # Read the picture  and turn it gray 
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            # Face recognition
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 4)
            
            # A loop for thr regen of interest 
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                
                # Appened both to features and labels to the variables 
                features.append(faces_roi)
                labels.append(label)
                
create_train()


print("trainning done ----------------")# Debugging Info

# Turn the features and labels list to arrays 
features = np.array(features , dtype='object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

# Make your own yml so you don't have to run the code everytime
face_recognizer.save('face_trained.yml')

# Saving them as a file using numpy.save 
np.save('features.npy', features)
np.save('labesl.npy', labels)