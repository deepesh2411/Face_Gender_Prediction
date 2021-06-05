from flask import render_template, request
from flask import redirect, url_for
from PIL import Image
import os
from app.utils import pipeline

UPLOAD_FOLDER = 'app/static/upload'


def base():
    return render_template("base.html")

def index():
    return render_template("index.html")

def faceapp():
    return render_template('faceapp.html')


def getwidth(path):
    img = Image.open(path)
    size = img.size #width height
    aspect = size[0] /size[1]   #width by height
    w = 250* aspect
    return int(w)

def gender():
    if request.method == 'POST':
        f = request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)
        w5 = getwidth(path)
        pipeline(path,filename,color='bgr')
        print("file saved to location {}".format(path))
        return render_template('gender.html',upload2=True, img_name=filename, w2=w5)

    return render_template('gender.html',upload2=False, img_name="deepesh.jpg", w2="300" )