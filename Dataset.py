import cv2
import numpy as np
import os
import pickle5 as pickle
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_data = []
i=0
name= input("Enter your name : ")
while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,scalefactor=1.3,minNeighbors=5)
    for (x,y,h,w) in faces:
        crop_img=frame[y:y+h,x:x+w,:]
        resized_img=cv2.resize(crop_img,dsize=(50,50))
        if len(face_data) <=100 and i%10 ==0:
            cv2.putText(frame,str(len(face_data)),org=(50,50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(50,50,255),thickness=2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
            
        cv2.imshow("frame",frame)
        k=cv2.waitKey(1)
        if k == 50:
            break
video.release()
cv2.destroyAllWindows()