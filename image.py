import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread("1.jpg")#load the image

result=DeepFace.analyze('1.jpg', enforce_detection=False)#analyse face


    #draw rectangle
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=facecascade.detectMultiScale(gray,1.1,4)
font=cv2.FONT_HERSHEY_SIMPLEX
i = 0
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(21,149,21),2)
    sex = 'Female'
    if result[0]['dominant_gender'] == 'Man':
        sex = 'male'
    output = f'{sex} {str(result[6-i]['age'])} {result[6-i]['dominant_emotion']}'
    cv2.putText(img,output,(x,y-15),font,0.5,(21,149,21),1,cv2.LINE_AA)
    print(result)
    i = i + 1
cv2.imshow("plot", img)
cv2.waitKey(0)
cv2.destroyAllWindows()