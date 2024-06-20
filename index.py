import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    ret , frame = web_Cam.read()
    
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    for (x, y, w, h) in faces_rect:
        
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness= 2)
        
    cv.imshow("Face Detection", frame)
    
    key = cv.waitKey(1)
     
    
    if key & 0xFF == ord('A'):
        break
    
web_Cam.release()
cv.destroyAllWindows()