# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:01:57 2020

@author: sumanth
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from detect import pre_dect
import time
import math
class VideoCamera(object):
    def __init__(self):
       self.prototxtPath = "./detector/deploy.prototxt"
       self.weightsPath ="./detector/res10_300x300_ssd_iter_140000.caffemodel"
       self.faceNet = cv2.dnn.readNet(self.prototxtPath,self.weightsPath)
       self.model=load_model('models/cnn_model')
       
       self.video = cv2.VideoCapture(0) 
    def __del__(self):
        cv2.destroyAllWindows()
        self.video.release()     
    def get_frame(self):
        try:
            ret, frame = self.video.read()
            now=time.time()
            (locs, preds)=pre_dect(frame,self.faceNet,self.model)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                cla=np.argmax(pred[0])
                label = "Mask" if cla==0 else "No Mask"
                color = (0, 255, 0) if cla == 0 else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(pred[0]) * 100)

            
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.line(frame,(startX,startY),(startX,startY+25),color,2)
                cv2.line(frame,(startX,startY),(startX+25,startY),color,2)
        
                cv2.line(frame,(endX,startY),(endX,startY+25),color,2)
                cv2.line(frame,(endX,startY),(endX-25,startY),color,2)
        
                cv2.line(frame,(startX,endY),(startX,endY-25),color,2)
                cv2.line(frame,(startX,endY),(startX+25,endY),color,2)
        
                cv2.line(frame,(endX, endY),(endX,endY-25),color,2)
                cv2.line(frame,(endX, endY),(endX-25,endY),color,2)
        
        
             #  cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            (hei, wid) = frame.shape[:2]
            #fps=cap.get(cv2.CAP_PROP_FPS)
            end=time.time()
            f=1/(end-now)
            FPS='FPS : '+str(math.ceil(f))
            cv2.putText(frame,str(FPS),(0,hei-20),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 1)
            no_faces='No. of faces in video   : '+str(len(locs))
            cv2.putText(frame,str(no_faces),(80,hei-20),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 1)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        except :
            pass