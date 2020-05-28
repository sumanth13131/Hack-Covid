# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:56:23 2020

@author: sumanth
"""

from flask import Flask, render_template, Response,request
from werkzeug.utils import secure_filename
from camera import VideoCamera
from imgdect import get_img
import cv2
#from detect import pre_dect
import os


app = Flask(__name__)
UPLOAD_FOLDER = '.\\imgssave'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template('Home.html')
@app.route('/image', methods=['POST','GET'])
def image():
    return render_template('image.html')
def img_gen():
    while True:
        frame=cv2.imread('./static/detectedimgs/detect.jpg')
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame=jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/img_feed')
def img_feed():
    return Response(img_gen(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/ShowImage',methods= ['GET', 'POST'])
def showimage():
    if request.method == 'POST':
        try:
            for imgs in os.listdir('./imgssave/'):
                f='./imgssave/'+imgs
                os.remove(f)
        except :
            pass
        image = request.files['myfile']
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        get_img()
        
        return render_template('imageshow.html')
    return render_template('image.html')

@app.route('/vedio')
def vedio():
    return render_template('vedio.html')


def ved_gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(ved_gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/technology')
def tech():
    return render_template('tech.html')
if __name__ == '__main__':
    app.run(debug=True,port='5000')