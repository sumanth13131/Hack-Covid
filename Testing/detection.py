# -*- coding: utf-8 -*-
"""
Created on Sat May  9 08:54:44 2020

@author: sumanth
"""
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import math
import time

prototxtPath = "./detector/deploy.prototxt"
weightsPath ="./detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model=load_model('models/cnn_model')

cap=cv2.VideoCapture(0)

def pre_dect(frame,faceNet,model):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

	
    for i in range(0, detections.shape[2]):
		
        confidence = detections[0, 0, i, 2]

		
        if confidence >=0.168:
			
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

			
            faces.append(face)
            locs.append((startX, startY, endX, endY))

	
    for k in faces:
        preds.append(model.predict(k))
    return (locs, preds)


while True:
    _,frame=cap.read()
    now=time.time()
    (locs, preds)=pre_dect(frame,faceNet,model)
    for (box, pred) in zip(locs, preds):
		
        (startX, startY, endX, endY) = box
        cla=np.argmax(pred[0])
        label = "Mask" if cla==0 else "No Mask"
        color = (0, 255, 0) if cla == 0 else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(pred[0]) * 100)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        cv2.line(frame,(startX,startY),(startX,startY+25),color,2)
        cv2.line(frame,(startX,startY),(startX+25,startY),color,2)
        
        cv2.line(frame,(endX,startY),(endX,startY+25),color,2)
        cv2.line(frame,(endX,startY),(endX-25,startY),color,2)
        
        cv2.line(frame,(startX,endY),(startX,endY-25),color,2)
        cv2.line(frame,(startX,endY),(startX+25,endY),color,2)
        
        cv2.line(frame,(endX, endY),(endX,endY-25),color,2)
        cv2.line(frame,(endX, endY),(endX-25,endY),color,2)
    (hei, wid) = frame.shape[:2]
    end=time.time()
    f=1/(end-now)
    FPS='FPS : '+str(math.ceil(f))
    no_faces='No. of faces in video   : '+str(len(locs))
    cv2.putText(frame,str(FPS),(0,hei-20),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 1)
    cv2.putText(frame,no_faces,(80,hei-20),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 1)
    cv2.imshow("Frame", frame)
    

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()