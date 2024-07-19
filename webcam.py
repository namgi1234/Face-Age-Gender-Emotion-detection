import cv2
from deepface import DeepFace

#open webcam
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Webcam")
#read face  
while True:
    ret,frame=cap.read()
    result=DeepFace.analyze(frame, enforce_detection=False)#analyse face
    #draw rectangle
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(gray,1.1,4)
    font=cv2.FONT_HERSHEY_SIMPLEX
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        sex = 'Female'
        if result[0]['dominant_gender'] == 'Man':
            sex = 'male'
        output = f'{sex} {str(result[0]['age'])} {result[0]['dominant_emotion']}'
        cv2.putText(frame,output,(x,y+10),font,1,(0,0,255),1,cv2.LINE_AA)
        print(result)
    #font type
    #write these things
    print(result.count('emotion'))
    # for i in range(result.count('emotion')):
        
    cv2.imshow('Demo video',frame)
    
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
